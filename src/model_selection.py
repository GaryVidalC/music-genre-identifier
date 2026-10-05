import json
import os
import pickle
import time
from pathlib import Path

import mlflow
import optuna
import pandas as pd
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

RANDOM_STATE = 42
TARGET_COLUMN = "genre"
MODELS = ["SVC", "RandomForest", "XGBClassifier"]
REGISTERED_MODEL_NAME = "music-genre-classifier"


def load_features(data_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load features parquet and split into predictors and target."""
    data = pd.read_parquet(data_path)
    features = data.drop(columns=[TARGET_COLUMN])
    target = data[TARGET_COLUMN]
    return features, target


def load_genre_encoder(encoder_path: Path):
    """Load genre encoder (LabelEncoder) from pickle file."""
    with encoder_path.open("rb") as file:
        encoder = pickle.load(file)
    return encoder


def build_model(model_name, params):
    """Build a supported classifier with the provided parameters.

    Args:
        model_name: Name of the model family to build.
        params: Hyperparameters used to configure the classifier.

    Returns:
        A configured classifier ready to be trained.
    """
    if model_name == "SVC":
        return make_pipeline(
            StandardScaler(),
            SVC(
                random_state=RANDOM_STATE,
                probability=True,
                **params,
            ),
        )

    if model_name == "RandomForest":
        return RandomForestClassifier(random_state=RANDOM_STATE, **params)

    if model_name == "XGBClassifier":
        return XGBClassifier(
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric="mlogloss",
            **params,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def model_options(
    trial: optuna.Trial,
    model_name: str,
) -> tuple[object, dict[str, object]]:
    """Sample hyperparameters and build a model for an Optuna trial.

    Args:
        trial: Optuna trial used to sample hyperparameters.
        model_name: Name of the model family to optimize.

    Returns:
        A tuple containing the configured model and sampled parameters.
    """
    if model_name == "SVC":
        C = trial.suggest_float("C", 0.1, 100, log=True)
        gamma = trial.suggest_categorical("gamma", ["scale", 0.01, 0.1, 1])
        params = {
            "C": C,
            "gamma": gamma,
        }

    elif model_name == "RandomForest":
        n_estimators = trial.suggest_int("n_estimators", 100, 200)
        max_depth = trial.suggest_int("max_depth", 10, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 6)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 4)
        max_features = trial.suggest_categorical(
            "max_features", ["sqrt", "log2"])
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
        }

    elif model_name == "XGBClassifier":
        learning_rate = trial.suggest_float("learning_rate", 0.01, 1.0)
        max_depth = trial.suggest_int("max_depth", 3, 10)
        params = {
            "learning_rate": learning_rate,
            "max_depth": max_depth,
        }
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return build_model(model_name, params), params


def evaluate_model(model, features, target):
    """Evaluate a trained model and measure its inference time.

    Args:
        model: Trained classifier to evaluate.
        features: Feature matrix used for prediction.
        target: Expected labels for the feature matrix.

    Returns:
        Accuracy, precision, recall, F1 score, and inference time.
    """
    start = time.perf_counter()
    predictions = model.predict(features)
    end = time.perf_counter()

    return {
        "accuracy": accuracy_score(target, predictions),
        "precision": precision_score(
            target,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "recall": recall_score(
            target,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "f1_score": f1_score(
            target,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "inference_time": end - start,
    }


def objective(trial, model_name, X_train, y_train, X_val, y_val):
    """Train and evaluate one Optuna trial inside a nested MLflow run.

    Args:
        trial: Optuna trial containing the sampled values.
        model_name: Name of the model family being optimized.
        X_train: Training feature matrix.
        y_train: Training labels.
        X_val: Validation feature matrix.
        y_val: Validation labels.

    Returns:
        Weighted F1 score used as the Optuna objective value.
    """
    with mlflow.start_run(
        nested=True,
        run_name=f"trial_{trial.number}",
    ) as child_run:
        model, params = model_options(trial, model_name)

        mlflow.log_params(params)
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_val, y_val)
        mlflow.log_metrics(metrics)
        trial.set_user_attr("run_id", child_run.info.run_id)

        return metrics["f1_score"]


def log_best_model(
    model: object,
    model_name: str,
    metadata: dict[str, object],
) -> object:
    """Log, register the final candidate model for one family.

    Args:
        model: Trained estimator to persist in MLflow.
        model_name: Model family used to choose its MLflow flavor.
        metadata: Training and preprocessing metadata stored with the model.

    Returns:
        Information about the model artifact and registered version.
    """

    if model_name == "RandomForest":
        model_info = mlflow.sklearn.log_model(
            model,
            name="model",
            serialization_format="skops",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
            registered_model_name=REGISTERED_MODEL_NAME,
            metadata=metadata,
        )
    elif model_name == "XGBClassifier":
        model_info = mlflow.xgboost.log_model(
            model,
            name="model",
            model_format="json",
            registered_model_name=REGISTERED_MODEL_NAME,
            metadata=metadata,
        )
    else:
        model_info = mlflow.sklearn.log_model(
            model,
            name="model",
            serialization_format="skops",
            registered_model_name=REGISTERED_MODEL_NAME,
            metadata=metadata,
        )

    return model_info


def main() -> None:
    """Optimize, train, evaluate, and register each model family."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # setting up MLflow tracking URI and experiment
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://localhost:5000",
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("model_selection")

    # loading data
    data_path = project_root / "processed_data/features.parquet"
    encoder_path = project_root / "model" / "genre_encoder.pkl"
    read_config_path = project_root / "configs" / "read_config.json"

    X, y = load_features(data_path)
    encoder = load_genre_encoder(encoder_path)

    with read_config_path.open("r", encoding="utf-8") as file:
        read_config = json.load(file)

    model_metadata = {
        "preprocessing": read_config,
        "genre_classes": encoder.classes_.tolist(),
        "feature_names": X.columns.tolist(),
        "number_of_samples": int(X.shape[0]),
        "number_of_features": int(X.shape[1]),
    }

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("Data loaded successfully.")
    models_f1 = {}
    model_versions = {}
    for model_name in MODELS:
        with mlflow.start_run(run_name=model_name):
            n_trials = 30
            mlflow.log_params({
                "n_trials": n_trials,
                "random_state": RANDOM_STATE,
            })

            sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
            study = optuna.create_study(
                direction="maximize",
                sampler=sampler,
            )
            study.optimize(
                lambda trial: objective(
                    trial,
                    model_name,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                ),
                n_trials=n_trials,
            )

            best_trial = study.best_trial
            mlflow.log_params(best_trial.params)
            mlflow.set_tags({
                "dataset": "GTZAN",
                "model_type": model_name,
            })

            if best_run_id := best_trial.user_attrs.get("run_id"):
                mlflow.log_param("best_child_run_id", best_run_id)

            final_model = build_model(model_name, best_trial.params)
            final_model.fit(X_train, y_train)

            final_metrics = evaluate_model(final_model, X_val, y_val)
            mlflow.log_metrics(final_metrics)

            model_info = log_best_model(
                final_model,
                model_name,
                model_metadata,
            )
            model_versions[model_name] = model_info.registered_model_version
            models_f1[model_name] = final_metrics["f1_score"]

    max_model_name = max(models_f1, key=models_f1.get)
    max_model_version = model_versions[max_model_name]

    # Assign the champion alias to the candidate with the highest F1 score.
    client = MlflowClient()
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="champion",
        version=max_model_version,
    )


if __name__ == "__main__":
    print("Starting model selection process...")
    main()
