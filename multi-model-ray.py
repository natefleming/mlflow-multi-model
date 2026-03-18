# Databricks notebook source

# MAGIC %md
# MAGIC # Many-Model Training with Ray on Databricks — Pickle Bundle Approach
# MAGIC
# MAGIC **Pattern:** Train one scikit-learn model per SKU using Ray Data's
# MAGIC `groupby().map_groups()` with Bayesian hyperparameter optimization via Optuna.
# MAGIC Each fitted model is pickled as raw bytes and returned as data — no MLflow
# MAGIC calls happen on Ray workers.  After training, all models are bundled into a
# MAGIC single MLflow pyfunc artifact.
# MAGIC
# MAGIC **How it works:**
# MAGIC 1. A Spark DataFrame is created and cached **before** the Ray cluster starts.
# MAGIC 2. `ray.data.from_spark()` converts it to a Ray Dataset (zero-copy Arrow).
# MAGIC 3. `groupby("sku_id").map_groups()` dispatches one function call per SKU to
# MAGIC    Ray workers in parallel.
# MAGIC 4. Inside each worker, Optuna runs Bayesian HPO (TPE sampler) to find the
# MAGIC    best RandomForest hyperparameters.
# MAGIC 5. The best model is pickled and returned as bytes — no MLflow logging on workers.
# MAGIC 6. After collecting results, the Ray cluster is shut down and Spark is used
# MAGIC    for Delta writes, MLflow logging, and inference.
# MAGIC
# MAGIC **Resource sharing between Spark and Ray:**
# MAGIC `setup_ray_cluster` creates long-running Spark tasks that occupy executor
# MAGIC slots.  Any Spark actions (`.count()`, `.collect()`, `.saveAsTable()`) will
# MAGIC hang if Ray holds all task slots.  This notebook avoids the issue by running
# MAGIC all Spark operations **before** or **after** the Ray cluster lifecycle.
# MAGIC
# MAGIC **Why pickle instead of per-SKU MLflow logging?**
# MAGIC Logging each SKU as its own MLflow run creates significant overhead at scale:
# MAGIC - Thousands of environment captures and redundant metadata
# MAGIC - Heavy artifact upload overhead and excessive run creation time
# MAGIC - MLflow API rate limiting under pressure
# MAGIC - An unusable MLflow experiment UI (too many runs to navigate or compare)
# MAGIC - Difficult inference workflows (loading models across thousands of runs)
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
import pickle

import mlflow
import mlflow.pyfunc
import optuna
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split

optuna.logging.set_verbosity(optuna.logging.WARNING)

from pyspark.sql import DataFrame
import pyspark.sql.functions as F
import pyspark.sql.types as T

# COMMAND ----------

# DBTITLE 1,Configuration

# Catalog and schema are configurable via Databricks widgets so users can
# override them from the notebook UI or pass values when running as a job:
#   dbutils.notebook.run("multi-model-ray", 0,
#       {"catalog_name": "prod", "schema_name": "ml"})
dbutils.widgets.text("catalog_name", "main", "Unity Catalog")
dbutils.widgets.text("schema_name", "forecasting", "Schema")

catalog_name: str = dbutils.widgets.get("catalog_name")
schema_name: str = dbutils.widgets.get("schema_name")
model_table: str = f"{catalog_name}.{schema_name}.sku_models_ray"
uc_model_name: str = f"{catalog_name}.{schema_name}.sku_model_ray"
bundle_volume_path: str = f"/Volumes/{catalog_name}/{schema_name}/model_artifacts"
feature_cols: list[str] = [
  "feat_1", "feat_2", "feat_3"
]
target_col: str = "sales"
num_tune_samples: int = 10

# Explicit schema for the results table.  Defining it up front prevents
# Spark's type inference from drifting between runs (e.g. inferring
# DoubleType vs FloatType for num_rows).
results_schema: T.StructType = T.StructType([
  T.StructField("sku_id", T.StringType(), True),
  T.StructField("model_bytes", T.BinaryType(), True),
  T.StructField("feature_names", T.ArrayType(T.StringType()), True),
  T.StructField("rmse", T.DoubleType(), True),
  T.StructField("mse", T.DoubleType(), True),
  T.StructField("mae", T.DoubleType(), True),
  T.StructField("num_rows", T.LongType(), True),
  T.StructField("n_estimators", T.LongType(), True),
  T.StructField("max_depth", T.LongType(), True),
  T.StructField("min_samples_split", T.LongType(), True),
  T.StructField("min_samples_leaf", T.LongType(), True),
  T.StructField("status", T.StringType(), True),
])

# COMMAND ----------

# DBTITLE 1,Initialize Unity Catalog and MLflow
spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog_name}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog_name}.{schema_name}")
spark.sql(f"USE {catalog_name}.{schema_name}")

spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog_name}.{schema_name}.model_artifacts")

mlflow.set_registry_uri("databricks-uc")
mlflow.autolog(disable=True)

# COMMAND ----------

# DBTITLE 1,Create synthetic training data (Spark — before Ray)

# All Spark operations must complete BEFORE setup_ray_cluster(), because
# Ray takes over Spark executor task slots.  Any Spark action (.count(),
# .collect(), .saveAsTable()) will hang while Ray holds those slots.
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

# Convert to pandas now while Spark executors are available.
# ray.data.from_spark() also needs executor slots and will hang after
# setup_ray_cluster.  For large datasets, use ray.data.read_databricks_tables()
# (reads via SQL warehouse, bypasses Spark executors entirely).
training_pdf: pd.DataFrame = df.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ray Cluster Lifecycle
# MAGIC
# MAGIC Everything between `setup_ray_cluster()` and `shutdown_ray_cluster()` runs
# MAGIC on Ray.  No Spark actions are allowed in this section — Ray occupies all
# MAGIC Spark executor task slots.

# COMMAND ----------

# DBTITLE 1,Configure Ray log collection path
from pathlib import Path

current_dir: Path = Path(
  dbutils.entry_point.getDbutils().notebook().getContext().notebookPath().get()
).parent
collect_log_to_path: Path = Path("/dbfs") / current_dir.relative_to("/") / "ray_collected_logs"
collect_log_to_path.as_posix()

# COMMAND ----------

# DBTITLE 1,Start Ray cluster
import ray
import ray.data
from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster

# Recommended Spark configs on the Databricks cluster:
#   spark.task.resource.gpu.amount 0   — reserve GPUs for Ray, not Spark tasks
#   RAY_memory_monitor_refresh_ms  0   — avoid spurious OOM kills from Ray
#
# num_cpus_per_node / num_gpus_per_node must be set together.
# For CPU-only sklearn workloads, set num_gpus_per_node=0.
# For GPU-accelerated models (XGBoost CUDA, LightGBM GPU), set to 1.
setup_ray_cluster(
  min_worker_nodes=2,
  max_worker_nodes=8,
  num_cpus_per_node=4,
  num_gpus_per_node=0,
  collect_log_to_path=collect_log_to_path.as_posix(),
)

ray.init(ignore_reinit_error=True)

print(f"Ray cluster resources: {ray.cluster_resources()}")

# COMMAND ----------

# DBTITLE 1,Create Ray Dataset from pandas

# We use from_pandas() instead of from_spark() because from_spark() also
# requires Spark executor slots (which Ray is holding).  For large datasets
# that don't fit in driver memory, save to a Delta table before Ray starts
# and use ray.data.read_databricks_tables() with a SQL warehouse.
ray_ds: ray.data.Dataset = ray.data.from_pandas(training_pdf)

print(f"Ray Dataset schema: {ray_ds.schema()}")
print(f"Ray Dataset count:  {ray_ds.count()}")

# COMMAND ----------

# DBTITLE 1,Define per-SKU tuner (Ray Data -> Optuna HPO -> Pickle)
def per_sku_tuner(group_df: pd.DataFrame,
                  columns_to_group: list[str],
                  num_trials: int) -> pd.DataFrame:
  """Run Bayesian HPO for one SKU group via Optuna, returning pickled model bytes.

  This function is called by ray.data.groupby().map_groups() — one
  invocation per SKU, executed in parallel across the Ray cluster.  Each
  worker runs Optuna's TPE sampler sequentially (no nested Ray task
  scheduling), avoiding the deadlock that occurs when nesting Ray Tune
  inside map_groups.

  No MLflow calls happen here — the fitted model is pickled and returned as
  raw bytes in the output DataFrame.  This eliminates worker-side auth
  requirements and MLflow API overhead during distributed training.

  Parallelism model:
    - OUTER: Ray Data dispatches N groups concurrently (one per worker).
    - INNER: Optuna runs num_trials sequentially within each worker.
    - TREE:  sklearn uses n_jobs=-1 to parallelize tree fitting across
             the CPU cores available on that Ray worker.

  Args:
    group_df:         Pandas DataFrame for one SKU (all rows share sku_id).
    columns_to_group: Column names used for grouping (["sku_id"]).
    num_trials:       Number of Optuna HPO trials per SKU.

  Returns:
    Single-row DataFrame with pickled model bytes, feature_names, best
    hyperparameters, and evaluation metrics.
  """
  import pickle as _pickle

  sku_id: str = str(group_df[columns_to_group[0]].iloc[0])

  try:
    X: pd.DataFrame = group_df.drop(columns=[target_col] + columns_to_group)
    y: pd.Series = group_df[target_col]
    feature_names: list[str] = list(X.columns)

    X_train, X_val, y_train, y_val = train_test_split(
      X, y, test_size=0.2, random_state=42
    )

    best_rmse: float = float("inf")
    best_config: dict = {}
    best_model: RandomForestRegressor | None = None

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

    model_bytes: bytes = _pickle.dumps(best_model, protocol=_pickle.HIGHEST_PROTOCOL)

    return pd.DataFrame([{
      "sku_id": sku_id,
      "model_bytes": model_bytes,
      "feature_names": feature_names,
      "rmse": final_rmse,
      "mse": final_mse,
      "mae": final_mae,
      "num_rows": len(group_df),
      "n_estimators": best_config.get("n_estimators"),
      "max_depth": best_config.get("max_depth"),
      "min_samples_split": best_config.get("min_samples_split"),
      "min_samples_leaf": best_config.get("min_samples_leaf"),
      "status": "success",
    }])

  except Exception as e:
    print(f"Error training model for sku_id: {sku_id}: {e}")

    return pd.DataFrame([{
      "sku_id": sku_id,
      "model_bytes": None,
      "feature_names": None,
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
  },
)

# Collect results to pandas while still on Ray — this is the last Ray operation.
results_pdf: pd.DataFrame = results_ds.to_pandas()

print(f"\nTraining complete: {len(results_pdf)} SKUs processed")
print(results_pdf[["sku_id", "rmse", "n_estimators", "max_depth"]].to_string(index=False))

# COMMAND ----------

# DBTITLE 1,Shutdown Ray cluster

# Shut down Ray to release Spark executor task slots.  All subsequent cells
# use Spark (Delta writes, MLflow logging, inference).
shutdown_ray_cluster()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Post-Training: Delta Writes, MLflow Logging, Inference
# MAGIC
# MAGIC Ray is shut down.  Spark executor slots are free again for Delta writes,
# MAGIC MLflow model logging, UC registration, and distributed inference.

# COMMAND ----------

# DBTITLE 1,Write results to Delta table

# Write the full results (including model_bytes) to a Delta table.
results_cols: list[str] = [
  "sku_id", "model_bytes", "feature_names",
  "rmse", "mse", "mae", "num_rows",
  "n_estimators", "max_depth", "min_samples_split", "min_samples_leaf",
  "status",
]
registry_df: DataFrame = spark.createDataFrame(results_pdf[results_cols], schema=results_schema)
registry_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(model_table)

display(
  spark.table(model_table)
  .select("sku_id", "rmse", "mse", "mae", "num_rows", F.length("model_bytes").alias("model_size_bytes"), "n_estimators", "max_depth", "status")
)

# COMMAND ----------

# DBTITLE 1,Bundle all pickled models into a single artifact

# Collect the pickled model bytes from the Delta table into a single dict
# keyed by sku_id.  This dict is itself pickled to produce one portable file
# that the custom pyfunc model loads at inference time.
model_pd: pd.DataFrame = spark.table(model_table).filter("status = 'success'").toPandas()

bundle: dict[str, dict] = {
  row["sku_id"]: {
    "model_bytes": row["model_bytes"],
    "feature_names": row["feature_names"],
  }
  for _, row in model_pd.iterrows()
}

bundle_path: str = f"{bundle_volume_path}/all_sku_models_ray.pkl"
with open(bundle_path, "wb") as f:
  pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

size_bytes: int = os.path.getsize(bundle_path)
print(f"Bundle contains {len(bundle)} models")
print(f"Pickle size: {size_bytes:,} bytes ({size_bytes / (1024 * 1024):.2f} MB)")

# COMMAND ----------

# DBTITLE 1,Save per-SKU metrics as CSV artifact

# Per-SKU evaluation metrics are saved as a CSV so they can be explored in
# a notebook or attached to the MLflow run as an artifact for later analysis.
metrics_df: pd.DataFrame = model_pd[[
  "sku_id", "rmse", "mse", "mae", "num_rows",
  "n_estimators", "max_depth", "min_samples_split", "min_samples_leaf",
]]
metrics_path: str = f"{bundle_volume_path}/sku_metrics_ray.csv"
metrics_df.to_csv(metrics_path, index=False)

print(f"Saved metrics for {len(metrics_df)} SKUs to {metrics_path}")
display(metrics_df)

# COMMAND ----------

# DBTITLE 1,Define the custom pyfunc model

class MultiSkuModel(mlflow.pyfunc.PythonModel):
  """A single MLflow model that contains all per-SKU models.

  At load time, the pickle bundle is deserialized into an in-memory dict
  mapping sku_id -> fitted sklearn model.  At predict time, the input
  DataFrame must contain an 'sku_id' column; each row is routed to the
  corresponding model for prediction.
  """

  def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
    import pickle as _pickle

    with open(context.artifacts["sku_models"], "rb") as f:
      raw: dict = _pickle.load(f)

    self.models: dict[str, dict] = {}
    for sku_id, entry in raw.items():
      model = _pickle.loads(entry["model_bytes"])
      self.models[sku_id] = {
        "model": model,
        "feature_names": entry["feature_names"],
      }

  def predict(self, context: mlflow.pyfunc.PythonModelContext, model_input: pd.DataFrame) -> pd.DataFrame:
    predictions: list[float] = []
    for _, row in model_input.iterrows():
      sku_id: str = row["sku_id"]
      entry = self.models.get(sku_id)
      if entry is None:
        raise KeyError(f"No model found for sku_id={sku_id}")

      feature_names: list[str] = entry["feature_names"]
      missing: list[str] = [c for c in feature_names if c not in row.index]
      if missing:
        raise KeyError(f"Missing feature(s) {missing} for sku_id={sku_id}")

      X: np.ndarray = row[feature_names].to_numpy().reshape(1, -1)
      predictions.append(entry["model"].predict(X)[0])

    return pd.DataFrame({"prediction": predictions})

# COMMAND ----------

# DBTITLE 1,Log bundle as a single MLflow run and register in UC

# One MLflow run wraps all per-SKU models.  This avoids the overhead of
# thousands of individual runs and produces a single model artifact that
# can be loaded as a Spark UDF for distributed inference.
input_example: pd.DataFrame = df.drop("sales").toPandas().sample(5, random_state=42)
input_example.reset_index(inplace=True, drop=True)

with mlflow.start_run(run_name="sku_models_bundled_ray") as run:
  run_id: str = run.info.run_id

  mlflow.pyfunc.log_model(
    artifact_path="model",
    python_model=MultiSkuModel(),
    artifacts={"sku_models": bundle_path},
    input_example=input_example,
  )

  mlflow.log_artifact(metrics_path, artifact_path="evaluations")

  avg_metrics: dict[str, float] = {
    "avg_rmse": float(metrics_df["rmse"].mean()),
    "avg_mse": float(metrics_df["mse"].mean()),
    "avg_mae": float(metrics_df["mae"].mean()),
    "num_skus": float(len(bundle)),
    "optuna_trials_per_sku": float(num_tune_samples),
  }
  mlflow.log_metrics(avg_metrics)

print(f"Logged model bundle under run_id: {run_id}")

# COMMAND ----------

# DBTITLE 1,Register in Unity Catalog
model_uri: str = f"runs:/{run_id}/model"
model_version = mlflow.register_model(model_uri, uc_model_name)

print(f"Registered {uc_model_name} version {model_version.version}")

# COMMAND ----------

# DBTITLE 1,Inference — Spark UDF (distributed)

# The pyfunc model routes each row to the correct SKU model internally.
# mlflow.pyfunc.spark_udf loads the bundle once per executor and applies
# predictions across the full DataFrame in parallel — no per-SKU model
# loading needed.
loaded_udf = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri)

inference_df: DataFrame = df.drop("sales")
predictions_df: DataFrame = inference_df.withColumn(
  "prediction", loaded_udf(F.struct(*[F.col(c) for c in inference_df.columns]))
)
display(predictions_df)

# COMMAND ----------

# DBTITLE 1,Inference — single-node pandas

# For small-batch or single-SKU inference, load the model directly.
model = mlflow.pyfunc.load_model(model_uri)

sample_input: pd.DataFrame = pd.DataFrame([
  {"sku_id": "SKU_001", "feat_1": 50.0, "feat_2": 25.0, "feat_3": 10.0},
  {"sku_id": "SKU_005", "feat_1": 75.0, "feat_2": 30.0, "feat_3": 15.0},
  {"sku_id": "SKU_010", "feat_1": 90.0, "feat_2": 40.0, "feat_3": 18.0},
])

predictions: pd.DataFrame = model.predict(sample_input)
result: pd.DataFrame = pd.concat([sample_input, predictions], axis=1)
print(result.to_string(index=False))

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
# MAGIC
# MAGIC Both notebooks use the pickle-bundle approach: workers return serialized model
# MAGIC bytes as data, and a single MLflow run wraps the entire model collection.
# MAGIC
# MAGIC **Rule of thumb:** Start with Spark `applyInPandas` for simplicity.  Move to
# MAGIC Ray when you need HPO per group, have 1000s+ groups, need GPU scheduling, or
# MAGIC hit Spark's per-task scheduling overhead.
