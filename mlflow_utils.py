"""
mlflow_utils.py
================
All MLflow helper functions for the Hand Gesture Recognition experiment.
Import this module in your notebook to keep tracking logic separate from
model logic.
"""

import os
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature


# ─────────────────────────────────────────────
# 1. Setup
# ─────────────────────────────────────────────

def setup_experiment(experiment_name: str = "hand-gesture-recognition") -> str:
    """
    Set (or create) the MLflow experiment.
    Always uses mlruns folder.
    Returns the experiment id.
    """
    tracking_uri = "mlruns"
    mlflow.set_tracking_uri(tracking_uri)

    if mlflow.active_run():
        mlflow.end_run()

    exp = mlflow.set_experiment(experiment_name)
    print(f"[MLflow] Tracking URI : {mlflow.get_tracking_uri()}")
    print(f"[MLflow] Experiment   : '{experiment_name}'  (id={exp.experiment_id})")
    return exp.experiment_id


# ─────────────────────────────────────────────
# 2. Dataset logging
# ─────────────────────────────────────────────

def log_dataset_info(df_raw, df_processed):
    """Log row/column counts and file paths for both datasets."""
    mlflow.log_param("raw_dataset_rows",       df_raw.shape[0])
    mlflow.log_param("raw_dataset_cols",       df_raw.shape[1])
    mlflow.log_param("processed_dataset_rows", df_processed.shape[0])
    mlflow.log_param("processed_dataset_cols", df_processed.shape[1])
    mlflow.log_param("num_classes",            df_raw["label"].nunique())

    processed_path = "processed_hand_landmarks_data.csv"
    if os.path.exists(processed_path):
        mlflow.log_artifact(processed_path, artifact_path="datasets")
        print(f"[MLflow] Dataset artifact logged: {processed_path}")


# ─────────────────────────────────────────────
# 3. Split info logging
# ─────────────────────────────────────────────

def log_split_info(x_train, x_val, x_test):
    """Log the sizes of train / val / test splits."""
    total = len(x_train) + len(x_val) + len(x_test)
    mlflow.log_param("train_size",    len(x_train))
    mlflow.log_param("val_size",      len(x_val))
    mlflow.log_param("test_size",     len(x_test))
    mlflow.log_param("train_pct",     round(len(x_train) / total, 2))
    mlflow.log_param("random_state",  100)
    mlflow.log_param("normalization", "wrist-origin / mid-finger-tip")


# ─────────────────────────────────────────────
# 4. Model parameters logging
# ─────────────────────────────────────────────

def log_model_params(model):
    """Log the sklearn model's hyperparameters with model prefix to avoid collisions."""
    params = model.get_params()
    model_type = type(model).__name__
    for k, v in params.items():
        mlflow.log_param(f"{model_type}_{k}", str(v)[:500])


# ─────────────────────────────────────────────
# 5. Metrics logging
# ─────────────────────────────────────────────

def log_metrics(metrics_dict: dict, prefix: str = "val"):
    """Log a flat dict of metric_name → float."""
    for name, value in metrics_dict.items():
        safe_name = f"{prefix}_{name}".replace(" ", "_").replace("\n", "_")
        mlflow.log_metric(safe_name, round(float(value), 6))


# ─────────────────────────────────────────────
# 6. Figure / artifact logging
# ─────────────────────────────────────────────

def log_figure(fig, filename: str, artifact_subdir: str = "plots"):
    """Save a matplotlib figure and log it as an MLflow artifact."""
    os.makedirs("tmp_artifacts", exist_ok=True)
    path = os.path.join("tmp_artifacts", filename)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    mlflow.log_artifact(path, artifact_path=artifact_subdir)
    print(f"[MLflow] Figure logged: {filename}")


# ─────────────────────────────────────────────
# 7. Model logging
# ─────────────────────────────────────────────

def log_model(model, model_name: str, x_sample):
    """Log a fitted sklearn model with its inferred signature."""
    signature = infer_signature(x_sample, model.predict(x_sample))
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        registered_model_name=None,
        input_example=x_sample.iloc[:3],
    )
    run_id = mlflow.active_run().info.run_id
    uri = f"runs:/{run_id}/model"
    print(f"[MLflow] Model logged → {uri}")
    return uri


# ─────────────────────────────────────────────
# 8. Model registry
# ─────────────────────────────────────────────

def register_model(run_id: str, registry_name: str = "hagrid-gesture-classifier"):
    """Register a model from a finished run into the MLflow Model Registry."""
    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=registry_name)
    print(f"[MLflow] Registered '{registry_name}'  version={mv.version}  run={run_id}")
    return mv


# ─────────────────────────────────────────────
# 9. Full run wrapper
# ─────────────────────────────────────────────

def run_experiment(
    run_name:    str,
    model,
    model_label: str,
    x_train, y_train,
    x_val,   y_val,
    x_test,  y_test,
    df_raw,  df_processed,
    metrics_fn,
    comparison_fig=None,
):
    """
    End-to-end helper: opens a FRESH run, logs everything, closes the run.
    Returns the MLflow run_id.
    """
    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("model_type", model_label)
        mlflow.set_tag("dataset",    "HaGRID-landmarks")

        log_dataset_info(df_raw, df_processed)
        log_split_info(x_train, x_val, x_test)
        log_model_params(model)

        val_metrics  = metrics_fn(model, x_val,  y_val)
        test_metrics = metrics_fn(model, x_test, y_test)
        log_metrics(val_metrics,  prefix="val")
        log_metrics(test_metrics, prefix="test")

        log_model(model, model_label, x_train)

        if comparison_fig is not None:
            log_figure(comparison_fig, f"comparison_{run_name}.png", "plots")

        run_id = run.info.run_id
        print(f"[MLflow] Run '{run_name}' finished  run_id={run_id}")

    return run_id
