# Databricks notebook source

# MAGIC %md
# MAGIC # Many-Model Training with Spark `applyInPandas` — Pickle Bundle Approach
# MAGIC
# MAGIC **Pattern:** Train one scikit-learn model per SKU using Spark's Pandas Function
# MAGIC API (`groupby.applyInPandas`), pickle each fitted model as raw bytes, and bundle
# MAGIC all models into a single MLflow artifact for unified inference.
# MAGIC
# MAGIC **How it works:**
# MAGIC 1. The Spark DataFrame is hash-partitioned by the group key (`sku_id`).
# MAGIC 2. `applyInPandas` collects each group into a pandas DataFrame on one executor
# MAGIC    (serialized via Apache Arrow) and calls the training function.
# MAGIC 3. The training function returns `model_bytes` (pickled model), `feature_names`,
# MAGIC    and evaluation metrics — **no MLflow calls happen on workers**.
# MAGIC 4. All pickled models are collected into a single bundle and logged as one
# MAGIC    MLflow pyfunc model, producing a single experiment run and one artifact.
# MAGIC
# MAGIC **Why pickle instead of per-SKU MLflow logging?**
# MAGIC Logging each SKU as its own MLflow run creates significant overhead at scale:
# MAGIC - Thousands of environment captures and redundant metadata
# MAGIC - Heavy artifact upload overhead and excessive run creation time
# MAGIC - MLflow API rate limiting under pressure
# MAGIC - An unusable MLflow experiment UI (too many runs to navigate or compare)
# MAGIC - Difficult inference workflows (loading models across thousands of runs)
# MAGIC
# MAGIC The pickle-bundle approach eliminates all of this: workers return model bytes
# MAGIC as data, and a single MLflow run wraps everything.
# MAGIC
# MAGIC **When to use:**
# MAGIC - Each group fits comfortably in single-executor memory.
# MAGIC - You need a *different* model per segment (per store, SKU, customer, etc.).
# MAGIC - No hyperparameter tuning is needed (or a simple grid search is sufficient).
# MAGIC
# MAGIC **See also:** `multi-model-ray.py` for a Ray-based approach with Bayesian HPO
# MAGIC via Optuna, which offers lower scheduling overhead and scales to 1M+ groups.

# COMMAND ----------

import pickle

import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, struct
import pyspark.sql.types as T

# COMMAND ----------

# DBTITLE 1,Configuration

# Catalog and schema are configurable via Databricks widgets so users can
# override them from the notebook UI or pass values when running as a job:
#   dbutils.notebook.run("multi-model-spark", 0,
#       {"catalog_name": "prod", "schema_name": "ml"})
dbutils.widgets.text("catalog_name", "main", "Unity Catalog")
dbutils.widgets.text("schema_name", "forecasting", "Schema")

notebook_path: str = dbutils.entry_point.getDbutils().notebook().getContext().notebookPath().get()

experiment_name: str = notebook_path
catalog_name: str = dbutils.widgets.get("catalog_name")
schema_name: str = dbutils.widgets.get("schema_name")
model_table: str = f"{catalog_name}.{schema_name}.sku_models"
uc_model_name: str = f"{catalog_name}.{schema_name}.sku_model"
bundle_volume_path: str = f"/Volumes/{catalog_name}/{schema_name}/model_artifacts"
feature_cols: list[str] = [
  "feat_1", "feat_2", "feat_3"
]
target_col: str = "sales"

# COMMAND ----------

# DBTITLE 1,Initialize Unity Catalog and MLflow experiment
spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog_name}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog_name}.{schema_name}")
spark.sql(f"USE {catalog_name}.{schema_name}")

# Ensure the UC Volume exists for storing the pickle bundle
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog_name}.{schema_name}.model_artifacts")

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(experiment_name)
mlflow.autolog(disable=True)

# COMMAND ----------

# DBTITLE 1,Define return schema and training function

# The return schema includes model_bytes (the pickled sklearn model) and
# feature_names (the columns used for training).  Every code path — success,
# insufficient data, error — must return a DataFrame matching this schema.
return_schema: T.StructType = T.StructType([
  T.StructField("sku_id", T.StringType(), True),
  T.StructField("model_bytes", T.BinaryType(), True),
  T.StructField("feature_names", T.ArrayType(T.StringType()), True),
  T.StructField("rmse", T.DoubleType(), True),
  T.StructField("mse", T.DoubleType(), True),
  T.StructField("mae", T.DoubleType(), True),
  T.StructField("num_rows", T.LongType(), True),
  T.StructField("status", T.StringType(), True),
])


def train_sku_model(pdf: pd.DataFrame) -> pd.DataFrame:
  """Train a RandomForest for a single SKU and return the pickled model.

  This function is called once per group by applyInPandas.  Spark guarantees
  that all rows share the same sku_id.  The function:
    1. Splits into train/validation sets to avoid overfitting metrics.
    2. Trains a RandomForestRegressor with n_jobs=-1 for multi-core fitting.
    3. Pickles the fitted model into bytes (no MLflow calls on the worker).
    4. Returns a single-row DataFrame with model_bytes, feature_names, and metrics.

  Because no MLflow logging happens here, there is no need for worker-side
  Databricks authentication.  This is a pure compute function.
  """
  sku_id: str = str(pdf["sku_id"].iloc[0])

  try:
    X: pd.DataFrame = pdf[feature_cols].fillna(0)
    y: pd.Series = pdf[target_col]

    if len(pdf) < 10:
      return pd.DataFrame([{
        "sku_id": sku_id,
        "model_bytes": None,
        "feature_names": None,
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

    model_bytes: bytes = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)

    return pd.DataFrame([{
      "sku_id": sku_id,
      "model_bytes": model_bytes,
      "feature_names": list(X.columns),
      "rmse": rmse,
      "mse": mse,
      "mae": mae,
      "num_rows": len(pdf),
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

# No parent MLflow run is needed — training is pure compute that returns
# pickled model bytes.  MLflow is only used later for the single bundle log.
num_skus: int = df.select("sku_id").distinct().count()
num_partitions: int = min(num_skus, spark.sparkContext.defaultParallelism * 2)

results_df: DataFrame = (
  df.repartition(num_partitions, "sku_id")
  .groupby("sku_id")
  .applyInPandas(train_sku_model, schema=return_schema)
)

results_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(model_table)

# COMMAND ----------

# DBTITLE 1,Review training results
display(
  spark.table(model_table)
  .select("sku_id", "rmse", "mse", "mae", "num_rows", "status")
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

bundle_path: str = f"{bundle_volume_path}/all_sku_models.pkl"
with open(bundle_path, "wb") as f:
  pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

import os
size_bytes: int = os.path.getsize(bundle_path)
print(f"Bundle contains {len(bundle)} models")
print(f"Pickle size: {size_bytes:,} bytes ({size_bytes / (1024 * 1024):.2f} MB)")

# COMMAND ----------

# DBTITLE 1,Save per-SKU metrics as CSV artifact

# Per-SKU evaluation metrics are saved as a CSV so they can be explored in
# a notebook or attached to the MLflow run as an artifact for later analysis.
metrics_df: pd.DataFrame = model_pd[["sku_id", "rmse", "mse", "mae", "num_rows"]]
metrics_path: str = f"{bundle_volume_path}/sku_metrics.csv"
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
    with open(context.artifacts["sku_models"], "rb") as f:
      raw: dict = pickle.load(f)

    self.models: dict[str, dict] = {}
    for sku_id, entry in raw.items():
      model = pickle.loads(entry["model_bytes"])
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

with mlflow.start_run(run_name="sku_models_bundled") as run:
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
  "prediction", loaded_udf(struct(*[col(c) for c in inference_df.columns]))
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
