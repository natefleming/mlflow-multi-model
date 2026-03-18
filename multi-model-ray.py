# Databricks notebook source

# MAGIC %md
# MAGIC # Many-Model Training with Ray on Databricks
# MAGIC
# MAGIC **Pattern:** Train one scikit-learn model per SKU using Ray Data's
# MAGIC `groupby().map_groups()` with Bayesian hyperparameter optimization via Optuna.
# MAGIC
# MAGIC **How it works:**
# MAGIC 1. A Spark DataFrame is converted to a Ray Dataset via `ray.data.from_spark()`
# MAGIC    (zero-copy Arrow transfer).
# MAGIC 2. `groupby("sku_id").map_groups()` dispatches one function call per SKU to
# MAGIC    Ray workers in parallel.  Ray's task scheduling overhead is ~milliseconds
# MAGIC    (vs ~seconds for Spark tasks), enabling efficient scaling to 1000s+ groups.
# MAGIC 3. Inside each worker, Optuna runs Bayesian HPO (Tree-structured Parzen
# MAGIC    Estimator) to find the best RandomForest hyperparameters.
# MAGIC 4. Each trial and the best model are logged to MLflow as nested runs.
# MAGIC
# MAGIC **Why Optuna instead of Ray Tune?**
# MAGIC Ray Tune's `Tuner.fit()` schedules its own Ray tasks internally.  Running it
# MAGIC *inside* a `map_groups` worker creates nested Ray task scheduling which
# MAGIC deadlocks — the outer task holds resources the inner trials need.  Optuna
# MAGIC runs in-process (no Ray task nesting) and still provides Bayesian search.
# MAGIC
# MAGIC **Why Ray instead of Spark `applyInPandas`?**
# MAGIC - Lower per-task scheduling overhead (~ms vs ~s) — critical at 1000s of SKUs.
# MAGIC - Dynamic resource allocation — Ray can autoscale workers mid-job.
# MAGIC - Native GPU scheduling — `num_gpus` per task for GPU-accelerated models.
# MAGIC - Proven at scale: Anyscale benchmarks show 1M models in 30 minutes.
# MAGIC
# MAGIC **Cluster requirements:**
# MAGIC - Multi-node cluster (Ray on Spark requires at least 2 Spark executors).
# MAGIC - CPU-only workers are sufficient for RandomForest.  Set `num_gpus_per_node`
# MAGIC   if using GPU-accelerated models (XGBoost, LightGBM with CUDA).
# MAGIC
# MAGIC **See also:** `multi-model-spark.py` for a simpler Spark-only approach
# MAGIC without HPO, suitable when you have fewer SKUs or don't need tuning.

# COMMAND ----------

import os
import time
from typing import List

import ray
import ray.data
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
import optuna
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from mlflow.models import infer_signature
from mlflow.utils.databricks_utils import get_databricks_host_creds

optuna.logging.set_verbosity(optuna.logging.WARNING)

from pyspark.sql import DataFrame
import pyspark.sql.types as T

# COMMAND ----------

# DBTITLE 1,Configuration

# Catalog and schema are configurable via Databricks widgets so users can
# override them from the notebook UI or pass values when running as a job:
#   dbutils.notebook.run("multi-model-ray", 0,
#       {"catalog_name": "prod", "schema_name": "ml"})
dbutils.widgets.text("catalog_name", "albertsons", "Unity Catalog")
dbutils.widgets.text("schema_name", "forecasting", "Schema")

notebook_path: str = dbutils.entry_point.getDbutils().notebook().getContext().notebookPath().get()

experiment_name: str = notebook_path
catalog_name: str = dbutils.widgets.get("catalog_name")
schema_name: str = dbutils.widgets.get("schema_name")
model_registry: str = f"{catalog_name}.{schema_name}.sku_model_registry_ray"
feature_cols: list[str] = [
  "feat_1", "feat_2", "feat_3"
]
target_col: str = "sales"
num_tune_samples: int = 10

# Explicit schema for the registry table.  Defining it up front prevents
# Spark's type inference from drifting between runs (e.g. inferring
# DoubleType vs FloatType for num_rows).
registry_schema: T.StructType = T.StructType([
  T.StructField("sku_id", T.StringType(), True),
  T.StructField("run_id", T.StringType(), True),
  T.StructField("rmse", T.DoubleType(), True),
  T.StructField("mse", T.DoubleType(), True),
  T.StructField("mae", T.DoubleType(), True),
  T.StructField("num_rows", T.LongType(), True),
  T.StructField("status", T.StringType(), True),
])

# COMMAND ----------

# DBTITLE 1,Initialize Unity Catalog and MLflow experiment
spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog_name}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog_name}.{schema_name}")
spark.sql(f"USE {catalog_name}.{schema_name}")

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(experiment_name)

# COMMAND ----------

# DBTITLE 1,Configure Ray log collection path
from pathlib import Path

current_dir: Path = Path(
  dbutils.entry_point.getDbutils().notebook().getContext().notebookPath().get()
).parent
collect_log_to_path: Path = Path("/dbfs") / current_dir.relative_to("/") / "ray_collected_logs"
collect_log_to_path.as_posix()

# COMMAND ----------

# DBTITLE 1,Set credentials and initialize Ray cluster

# Ray workers are separate processes on different cluster nodes.  They do
# NOT inherit the driver's Databricks authentication context.  We capture
# the credentials here and pass them explicitly to each map_groups worker
# via fn_kwargs (see the map_groups call below).
#
# Recommended Spark configs on the Databricks cluster:
#   spark.task.resource.gpu.amount 0   — reserve GPUs for Ray, not Spark tasks
#   RAY_memory_monitor_refresh_ms  0   — avoid spurious OOM kills from Ray
from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster

creds = get_databricks_host_creds()
os.environ["DATABRICKS_HOST"] = creds.host
os.environ["DATABRICKS_TOKEN"] = creds.token

# min/max_worker_nodes enables Ray autoscaling on Databricks.
# num_cpus_per_node / num_gpus_per_node should match your instance type.
# For CPU-only sklearn workloads, set num_gpus_per_node=0.
# For GPU-accelerated models (XGBoost CUDA, LightGBM GPU), set to 1.
setup_ray_cluster(
  min_worker_nodes=2,
  max_worker_nodes=8,
  num_cpus_per_node=4,
  collect_log_to_path=collect_log_to_path.as_posix(),
)

ray.init(ignore_reinit_error=True)

print(f"Ray cluster resources: {ray.cluster_resources()}")

# COMMAND ----------

# DBTITLE 1,Create synthetic training data (Spark DataFrame)
np.random.seed(42)

data: list[dict] = []
for sku_num in range(1, 11):
    sku_id: str = f"SKU_{sku_num:03d}"
    n_samples: int = 100

    feat_1: np.ndarray = np.random.uniform(10, 100, n_samples)
    feat_2: np.ndarray = np.random.uniform(5, 50, n_samples)
    feat_3: np.ndarray = np.random.uniform(1, 20, n_samples)

    sales: np.ndarray = (
        2.5 * feat_1 +
        1.8 * feat_2 +
        3.2 * feat_3 +
        np.random.normal(0, 20, n_samples)
    )

    for i in range(n_samples):
        data.append({
            "sku_id": sku_id,
            "feat_1": feat_1[i],
            "feat_2": feat_2[i],
            "feat_3": feat_3[i],
            "sales": max(0, sales[i]),
        })

df: DataFrame = spark.createDataFrame(pd.DataFrame(data))

print(f"Created dataset with {df.count()} total rows")
print(f"Number of unique SKUs: {df.select('sku_id').distinct().count()}")
print(f"Rows per SKU: {df.groupBy('sku_id').count().select('count').distinct().collect()[0][0]}")
print("\nSample data:")
display(df.limit(10))

# COMMAND ----------

# DBTITLE 1,Convert Spark DataFrame to Ray Dataset

# ray.data.from_spark() transfers data via Apache Arrow (zero-copy columnar
# format).  The resulting Ray Dataset is partitioned across the Ray object
# store and can be operated on without further Spark involvement.
ray_ds: ray.data.Dataset = ray.data.from_spark(df)

print(f"Ray Dataset schema: {ray_ds.schema()}")
print(f"Ray Dataset count:  {ray_ds.count()}")

# COMMAND ----------

# DBTITLE 1,Define per-SKU tuner (Ray Data -> Optuna HPO -> MLflow)
def per_sku_tuner(group_df: pd.DataFrame,
                  columns_to_group: list[str],
                  num_trials: int,
                  databricks_host: str,
                  databricks_token: str) -> pd.DataFrame:
  """Run Bayesian HPO for one SKU group via Optuna, logging to MLflow.

  This function is called by ray.data.groupby().map_groups() — one
  invocation per SKU, executed in parallel across the Ray cluster.  Each
  worker runs Optuna's TPE sampler sequentially (no nested Ray task
  scheduling), avoiding the deadlock that occurs when nesting Ray Tune
  inside map_groups.

  Parallelism model:
    - OUTER: Ray Data dispatches N groups concurrently (one per worker).
    - INNER: Optuna runs num_trials sequentially within each worker.
    - TREE:  sklearn uses n_jobs=-1 to parallelize tree fitting across
             the CPU cores available on that Ray worker.

  MLflow run hierarchy:
    parent_run (group_{sku_id})
      ├── trial_0_{sku_id}   (nested — logged by objective())
      ├── trial_1_{sku_id}
      ├── ...
      └── BEST_{sku_id}      (nested — best model artifact logged here)

  Args:
    group_df:         Pandas DataFrame for one SKU (all rows share sku_id).
    columns_to_group: Column names used for grouping (["sku_id"]).
    num_trials:       Number of Optuna HPO trials per SKU.
    databricks_host:  Workspace URL for worker-side MLflow auth.
    databricks_token: PAT for worker-side MLflow auth.

  Returns:
    Single-row DataFrame with best model metadata and metrics.
  """
  os.environ["DATABRICKS_HOST"] = databricks_host
  os.environ["DATABRICKS_TOKEN"] = databricks_token
  mlflow.set_tracking_uri("databricks")
  mlflow.set_registry_uri("databricks-uc")
  mlflow.set_experiment(experiment_name)

  sku_id: str = str(group_df[columns_to_group[0]].iloc[0])

  try:
    X: pd.DataFrame = group_df.drop(columns=[target_col] + columns_to_group)
    y: pd.Series = group_df[target_col]
    X_train, X_val, y_train, y_val = train_test_split(
      X, y, test_size=0.2, random_state=42
    )

    best_rmse: float = float("inf")
    best_config: dict = {}
    best_model: RandomForestRegressor | None = None

    with mlflow.start_run(run_name=f"group_{sku_id}") as parent_run:

      def objective(trial: optuna.Trial) -> float:
        nonlocal best_rmse, best_config, best_model

        config: dict = {
          "n_estimators": trial.suggest_int("n_estimators", 50, 300),
          "max_depth": trial.suggest_categorical("max_depth", [5, 10, 15, 20, None]),
          "min_samples_split": trial.suggest_categorical("min_samples_split", [2, 5, 10]),
          "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [1, 2, 4]),
        }

        model = RandomForestRegressor(**config, n_jobs=-1, random_state=42)
        model.fit(X_train, y_train)
        preds: np.ndarray = model.predict(X_val)

        rmse: float = float(root_mean_squared_error(y_val, preds))
        mse: float = float(mean_squared_error(y_val, preds))
        mae: float = float(mean_absolute_error(y_val, preds))

        with mlflow.start_run(run_name=f"trial_{trial.number}_{sku_id}", nested=True):
          mlflow.log_params({**config, "sku_id": sku_id})
          mlflow.log_metrics({"rmse": rmse, "mse": mse, "mae": mae})

        if rmse < best_rmse:
          best_rmse = rmse
          best_config = config
          best_model = model

        return rmse

      study: optuna.Study = optuna.create_study(
        direction="minimize", study_name=f"tune_{sku_id}"
      )
      study.optimize(objective, n_trials=num_trials)

      best_preds: np.ndarray = best_model.predict(X_val)
      final_rmse: float = float(root_mean_squared_error(y_val, best_preds))
      final_mse: float = float(mean_squared_error(y_val, best_preds))
      final_mae: float = float(mean_absolute_error(y_val, best_preds))

      signature = infer_signature(X_val.head(5), best_preds[:5])

      with mlflow.start_run(run_name=f"BEST_{sku_id}", nested=True) as best_run:
        mlflow.log_params({**best_config, "sku_id": sku_id})
        mlflow.log_metrics({
          "rmse": final_rmse,
          "mse": final_mse,
          "mae": final_mae,
          "num_rows": float(len(group_df)),
        })
        mlflow.sklearn.log_model(best_model, artifact_path="model", signature=signature)

    return pd.DataFrame([{
      "sku_id": sku_id,
      "run_id": best_run.info.run_id,
      "experiment_id": best_run.info.experiment_id,
      "rmse": final_rmse,
      "mse": final_mse,
      "mae": final_mae,
      "num_rows": len(group_df),
      "n_estimators": best_config["n_estimators"],
      "max_depth": best_config["max_depth"],
      "min_samples_split": best_config["min_samples_split"],
      "min_samples_leaf": best_config["min_samples_leaf"],
      "status": "success",
    }])

  except Exception as e:
    print(f"Error training model for sku_id: {sku_id}")
    print(e)

    return pd.DataFrame([{
      "sku_id": sku_id,
      "run_id": None,
      "experiment_id": None,
      "rmse": None,
      "mse": None,
      "mae": None,
      "num_rows": len(group_df),
      "n_estimators": None,
      "max_depth": None,
      "min_samples_split": None,
      "min_samples_leaf": None,
      "status": f"error: {e}",
    }])

# COMMAND ----------

# DBTITLE 1,Run distributed per-SKU HPO via Ray Data map_groups

# Auto-compute group_concurrency from available cluster CPUs so the notebook
# scales without manual tuning.  Each concurrent group occupies roughly one
# CPU (Optuna runs sequentially within each group).
resources: dict = ray.cluster_resources()
total_cluster_cpus: float = resources.get("CPU", 1.0)
group_concurrency: int = max(1, int(total_cluster_cpus))

print(f"Cluster CPUs: {total_cluster_cpus}")
print(f"Group concurrency (auto): {group_concurrency}")
print(f"Optuna trials per SKU: {num_tune_samples}")

results_ds: ray.data.Dataset = ray_ds.groupby("sku_id").map_groups(
  per_sku_tuner,
  concurrency=group_concurrency,
  fn_kwargs={
    "columns_to_group": ["sku_id"],
    "num_trials": num_tune_samples,
    "databricks_host": os.environ["DATABRICKS_HOST"],
    "databricks_token": os.environ["DATABRICKS_TOKEN"],
  },
)

results_pdf: pd.DataFrame = results_ds.to_pandas()

print(f"\nTraining complete: {len(results_pdf)} SKUs processed")
print(results_pdf[["sku_id", "rmse", "n_estimators", "max_depth"]].to_string(index=False))

# COMMAND ----------

# DBTITLE 1,Write registry table
registry_df: DataFrame = spark.createDataFrame(
  results_pdf[["sku_id", "run_id", "rmse", "mse", "mae", "num_rows", "status"]],
  schema=registry_schema,
)
registry_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(model_registry)

display(spark.table(model_registry))

# COMMAND ----------

# DBTITLE 1,Register models in Unity Catalog with aliases
def register_models(
  run_ids: list[dict],
  model_name: str,
  batch_size: int = 10,
  sleep_seconds: int = 2,
  max_retries: int = 3,
) -> list[dict]:
  """Register trained models into Unity Catalog and assign per-SKU aliases.

  Each model version gets:
    - A sku_id tag for filtering in the UC Model Registry UI.
    - An alias (e.g. "sku-001") so inference code can load by alias instead
      of hard-coding version numbers.

  Batching with sleep prevents rate-limit errors from the UC REST API when
  registering hundreds of models.
  """
  client: mlflow.tracking.MlflowClient = mlflow.tracking.MlflowClient()

  results: list[dict] = []

  for i in range(0, len(run_ids), batch_size):
    batch: list[dict] = run_ids[i:i+batch_size]

    for item in batch:
      sku_id: str = item["sku_id"]
      run_id: str = item["run_id"]

      for attempt in range(max_retries):
        try:
          model_uri: str = f"runs:/{run_id}/model"
          model_version = mlflow.register_model(model_uri, model_name)
          client.set_model_version_tag(
            name=model_name,
            version=model_version.version,
            key="sku_id",
            value=sku_id,
          )
          alias: str = sku_id.lower().replace("_", "-")
          client.set_registered_model_alias(
            name=model_name,
            alias=alias,
            version=model_version.version,
          )
          results.append({
            "sku_id": sku_id,
            "model_version": model_version.version,
            "alias": alias,
            "status": "registered",
          })
          break
        except Exception as e:
          if attempt == max_retries - 1:
            print(f"Error registering model for sku_id: {sku_id}")
            print(e)
            results.append({
              "sku_id": sku_id,
              "model_version": None,
              "status": f"failed: {e}",
            })
          else:
            time.sleep(sleep_seconds ** attempt)

    time.sleep(sleep_seconds)

  return results

# COMMAND ----------

# DBTITLE 1,Register top-performing models
top_skus: list[dict] = (
  spark.read.table(model_registry)
  .filter("status = 'success' AND rmse < 50.0")
  .select("sku_id", "run_id")
  .toPandas()
  .to_dict(orient="records")
)

print(f"Registering {len(top_skus)} models")

uc_model: str = f"{catalog_name}.{schema_name}.sku_model_ray"

results: list[dict] = register_models(
  run_ids=top_skus,
  model_name=uc_model,
  batch_size=10,
  sleep_seconds=2,
)

display(spark.createDataFrame(results))

# COMMAND ----------

# DBTITLE 1,Inference example using model aliases

# Load models by alias (e.g. "sku-001") rather than version number.
# This decouples inference code from model version bookkeeping — when a
# new model is registered and the alias is reassigned, inference
# automatically picks up the latest version.
sample_skus: list[str] = ["SKU_001", "SKU_005", "SKU_010"]

for sku_id in sample_skus:
  alias: str = sku_id.lower().replace("_", "-")
  model_uri: str = f"models:/{uc_model}@{alias}"
  model = mlflow.pyfunc.load_model(model_uri)

  sample_input: pd.DataFrame = pd.DataFrame([{
    "feat_1": np.random.uniform(10, 100),
    "feat_2": np.random.uniform(5, 50),
    "feat_3": np.random.uniform(1, 20),
  }])

  prediction: np.ndarray = model.predict(sample_input)
  print(f"SKU: {sku_id} (alias: {alias}) | Input: {sample_input.values[0]} | Predicted Sales: {prediction[0]:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Spark vs Ray: When to Use Each Approach
# MAGIC
# MAGIC | Dimension | Spark `applyInPandas` | Ray `map_groups` |
# MAGIC | --- | --- | --- |
# MAGIC | **Best for** | Simple per-group training, no HPO | Per-group training with HPO |
# MAGIC | **Scheduling overhead** | ~seconds per task | ~milliseconds per task |
# MAGIC | **Max practical groups** | 100s – 1000s | 1000s – 1M+ |
# MAGIC | **HPO integration** | Manual loop or Hyperopt | Optuna, Ray Tune |
# MAGIC | **GPU support** | Limited (Spark GPU scheduling) | Native (`num_gpus` per task) |
# MAGIC | **Data transfer** | Arrow (Spark to Python) | Arrow (Spark to Ray, zero-copy) |
# MAGIC | **Autoscaling** | Databricks autoscale (node-level) | Ray autoscale (task-level) |
# MAGIC | **Infrastructure** | Spark cluster only | Spark + Ray (additional setup) |
# MAGIC | **MLflow bottleneck** | `log_model()` per group | Same |
# MAGIC
# MAGIC **Rule of thumb:** Start with Spark `applyInPandas` for simplicity.  Move to
# MAGIC Ray when you need HPO per group, have 1000s+ groups, need GPU scheduling, or
# MAGIC hit Spark's per-task scheduling overhead.

# COMMAND ----------

# DBTITLE 1,Shutdown Ray cluster
shutdown_ray_cluster()
