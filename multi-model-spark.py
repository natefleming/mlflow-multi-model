# Databricks notebook source

# MAGIC %md
# MAGIC # Many-Model Training with Spark `applyInPandas`
# MAGIC
# MAGIC **Pattern:** Train one scikit-learn model per SKU using Spark's Pandas Function
# MAGIC API (`groupby.applyInPandas`).  This is the simplest Spark-native approach to
# MAGIC many-model training — no additional frameworks (Ray, Dask) are required.
# MAGIC
# MAGIC **How it works:**
# MAGIC 1. The Spark DataFrame is hash-partitioned by the group key (`sku_id`).
# MAGIC 2. `applyInPandas` collects each group into a pandas DataFrame on one executor
# MAGIC    (serialized via Apache Arrow) and calls your training function.
# MAGIC 3. The function returns a summary DataFrame; Spark reassembles all results.
# MAGIC
# MAGIC **When to use:**
# MAGIC - Each group fits comfortably in single-executor memory.
# MAGIC - You need a *different* model per segment (per store, SKU, customer, etc.).
# MAGIC - No hyperparameter tuning is needed (or a simple grid search is sufficient).
# MAGIC
# MAGIC **Limitations at scale (1000s+ SKUs):**
# MAGIC - `mlflow.sklearn.log_model()` is called once per SKU, which serializes and
# MAGIC   uploads the model artifact.  At 1000s of SKUs this dominates wall-clock time.
# MAGIC - Each group runs on *one* executor — there is no intra-group parallelism
# MAGIC   beyond what scikit-learn provides via `n_jobs`.
# MAGIC - Skewed group sizes can cause straggler executors.
# MAGIC
# MAGIC **See also:** `multi-model-ray.py` for a Ray-based approach with Bayesian HPO
# MAGIC via Optuna, which offers lower scheduling overhead and scales to 1M+ groups.

# COMMAND ----------

import time

import mlflow
import mlflow.sklearn
import mlflow.pyfunc
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from mlflow.models import infer_signature
from mlflow.utils.databricks_utils import get_databricks_host_creds

from pyspark.sql import DataFrame
import pyspark.sql.types as T

# COMMAND ----------

# DBTITLE 1,Configuration

# Catalog and schema are configurable via Databricks widgets so users can
# override them from the notebook UI or pass values when running as a job:
#   dbutils.notebook.run("multi-model-spark", 0,
#       {"catalog_name": "prod", "schema_name": "ml"})
dbutils.widgets.text("catalog_name", "albertsons", "Unity Catalog")
dbutils.widgets.text("schema_name", "forecasting", "Schema")

notebook_path: str = dbutils.entry_point.getDbutils().notebook().getContext().notebookPath().get()

experiment_name: str = notebook_path
catalog_name: str = dbutils.widgets.get("catalog_name")
schema_name: str = dbutils.widgets.get("schema_name")
model_registry: str = f"{catalog_name}.{schema_name}.sku_model_registry"
feature_cols: list[str] = [
  "feat_1", "feat_2", "feat_3"
]
target_col: str = "sales"

# COMMAND ----------

# DBTITLE 1,Initialize Unity Catalog and MLflow experiment
spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog_name}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog_name}.{schema_name}")
spark.sql(f"USE {catalog_name}.{schema_name}")

# COMMAND ----------

# Point the MLflow model registry at Unity Catalog (three-level names).
# set_experiment uses the notebook path so runs appear in the notebook's
# experiment sidebar.
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(experiment_name)

# COMMAND ----------

# DBTITLE 1,Capture Databricks credentials for Spark workers
# Spark executors are separate JVM/Python processes that do NOT inherit the
# driver's Databricks authentication context.  We capture host + token here
# on the driver and inject them into each worker via os.environ inside the
# applyInPandas function.  Without this, mlflow.start_run() on workers fails
# with "cannot configure default credentials".
creds = get_databricks_host_creds()
db_host: str = creds.host
db_token: str = creds.token

# COMMAND ----------

# DBTITLE 1,Define return schema and training function

# The return schema must be declared upfront because Spark needs it before
# execution begins (lazy evaluation).  Every code path in the UDF — success,
# insufficient data, error — must return a DataFrame matching this schema.
return_schema: T.StructType = T.StructType([
  T.StructField("sku_id", T.StringType(), True),
  T.StructField("run_id", T.StringType(), True),
  T.StructField("rmse", T.DoubleType(), True),
  T.StructField("mse", T.DoubleType(), True),
  T.StructField("mae", T.DoubleType(), True),
  T.StructField("num_rows", T.LongType(), True),
  T.StructField("status", T.StringType(), True),
])


def train_sku_model(pdf: pd.DataFrame) -> pd.DataFrame:
  """Train a RandomForest for a single SKU and log to MLflow.

  This function is called once per group by applyInPandas.  Spark guarantees
  that all rows share the same sku_id.  The function:
    1. Sets Databricks auth on the worker (env vars).
    2. Splits into train/validation sets to avoid overfitting metrics.
    3. Trains a RandomForestRegressor with n_jobs=-1 for multi-core fitting.
    4. Logs params, metrics, and the serialized model to a nested MLflow run.
    5. Returns a single-row summary DataFrame matching return_schema.

  At scale (1000s of SKUs), the dominant cost is mlflow.sklearn.log_model()
  which serializes + uploads the model artifact per SKU.
  """
  import os
  os.environ["DATABRICKS_HOST"] = db_host
  os.environ["DATABRICKS_TOKEN"] = db_token

  sku_id: str = str(pdf["sku_id"].iloc[0])

  try:
    X: pd.DataFrame = pdf[feature_cols].fillna(0)
    y: pd.Series = pdf[target_col]

    if len(pdf) < 10:
      return pd.DataFrame([{
        "sku_id": sku_id,
        "run_id": None,
        "rmse": None,
        "mse": None,
        "mae": None,
        "num_rows": len(pdf),
        "status": "insufficient_data",
      }])

    X_train, X_val, y_train, y_val = train_test_split(
      X, y, test_size=0.2, random_state=42
    )

    n_estimators: int = 50

    model = RandomForestRegressor(
      n_estimators=n_estimators,
      n_jobs=-1,
      random_state=42,
    )
    model.fit(X_train, y_train)
    preds: np.ndarray = model.predict(X_val)

    mse: float = float(mean_squared_error(y_val, preds))
    rmse: float = float(root_mean_squared_error(y_val, preds))
    mae: float = float(mean_absolute_error(y_val, preds))

    with mlflow.start_run(
      run_name=f"sku_{sku_id}",
      experiment_id=experiment_id,
      nested=True,
      tags={
        "sku_id": sku_id,
        "parent_run_id": parent_run_id,
      }
    ) as run:
      mlflow.log_params({"sku_id": sku_id, "n_estimators": n_estimators})
      mlflow.log_metrics({"mse": mse, "rmse": rmse, "mae": mae, "num_rows": float(len(pdf))})
      signature = infer_signature(X_val, preds)
      mlflow.sklearn.log_model(model, artifact_path="model", signature=signature)
      run_id: str = run.info.run_id

    return pd.DataFrame([{
      "sku_id": sku_id,
      "run_id": run_id,
      "rmse": rmse,
      "mse": mse,
      "mae": mae,
      "num_rows": len(pdf),
      "status": "success",
    }])
  except Exception as e:
    print(f"Error training model for sku_id: {sku_id}")
    print(e)

    return pd.DataFrame([{
      "sku_id": sku_id,
      "run_id": None,
      "rmse": None,
      "mse": None,
      "mae": None,
      "num_rows": len(pdf),
      "status": f"error: {e}",
    }])


# COMMAND ----------

# DBTITLE 1,Create synthetic training data
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

# DBTITLE 1,Train one model per SKU via applyInPandas

# A parent MLflow run groups all per-SKU child runs in the experiment UI.
# parent_run_id and experiment_id are closure-captured by train_sku_model.
with mlflow.start_run(run_name="sku_training_spark") as parent_run:

  parent_run_id: str = parent_run.info.run_id
  experiment_id: str = parent_run.info.experiment_id
  print(f"parent_run_id: {parent_run_id}")
  print(f"experiment_id: {experiment_id}")

  # Scale num_partitions with cluster size so each partition maps to one
  # Spark task.  For 1000s of SKUs use defaultParallelism * 2; for small
  # demo datasets cap at the number of distinct SKUs.
  num_skus: int = df.select("sku_id").distinct().count()
  num_partitions: int = min(num_skus, spark.sparkContext.defaultParallelism * 2)

  results_df: DataFrame = (
    df.repartition(num_partitions, "sku_id")
    .groupby("sku_id")
    .applyInPandas(train_sku_model, schema=return_schema)
  )

  results_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(model_registry)


# COMMAND ----------

# DBTITLE 1,Review training results
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

uc_model: str = f"{catalog_name}.{schema_name}.sku_model"

results: list[dict] = register_models(
  run_ids=top_skus,
  model_name=uc_model,
  batch_size=10,
  sleep_seconds=2,
)

display(spark.createDataFrame(results))

# COMMAND ----------

# DBTITLE 1,Inference example using model aliases

# At inference time, load the model by alias instead of version number.
# The alias ("sku-001") was set during registration and points to the
# latest registered version for that SKU.  This decouples inference code
# from model version bookkeeping.
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
