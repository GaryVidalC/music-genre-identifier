import pickle
import time
import pandas as pd
import mlflow
import optuna
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sqlalchemy import null
from xgboost import XGBClassifier

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("model_selection")

RANDOM_STATE = 42
TARGET_COLUMN = "genre"
MODELS = ["SVC", "RandomForest", "XGBClassifier"]


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

def model_options(trial, model_name):
    if model_name == "SVC":
        C = trial.suggest_float("C", 0.1, 100, log=True)
        gamma = trial.suggest_categorical("gamma", ["scale", 0.01, 0.1, 1])
        params = {
            "C": C,
            "gamma": gamma,
        }
        model = make_pipeline(StandardScaler(), SVC(random_state=RANDOM_STATE, probability=True, **params))

    elif model_name == "RandomForest":
        n_estimators = trial.suggest_int("n_estimators", 100, 200)
        max_depth = trial.suggest_int("max_depth", 10, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 6)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 4)
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
        }
        model = RandomForestClassifier(random_state=RANDOM_STATE, **params)

    elif model_name == "XGBClassifier":
        learning_rate = trial.suggest_float("learning_rate", 0.01, 1.0)
        max_depth = trial.suggest_int("max_depth", 3, 10)
        params = {
            "learning_rate": learning_rate,
            "max_depth": max_depth,
        }
        model = XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='mlogloss', **params)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model, params

def objective(trial, model_name, X_train, y_train, X_val, y_val):
    with mlflow.start_run(nested = True, run_name = f"trial_{trial.number}") as child_run:

        model, params =model_options(trial, model_name)
        mlflow.log_params(params)
        model.fit(X_train, y_train)

        start = time.perf_counter()
        y_pred = model.predict(X_val)
        end = time.perf_counter()

        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_val, y_pred, average='weighted', zero_division=0),
            "f1_score": f1_score(y_val, y_pred, average='weighted', zero_division=0),
            "inference_time": end - start
        }
        mlflow.log_metrics(metrics)
        if model_name == "RandomForest":
            mlflow.sklearn.log_model(
                model,
                name="model",
                serialization_format="skops",
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )
        elif model_name == "XGBClassifier":
            mlflow.xgboost.log_model(
                model,
                name="model",
                model_format="json",
            )
        else:
            mlflow.sklearn.log_model(
                model,
                name="model",
                serialization_format="skops",
            )
        trial.set_user_attr("run_id", child_run.info.run_id)
        return metrics["f1_score"]

def main() -> None:
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    data_path = project_root / "processed_data/features.parquet"
    X, y = load_features(data_path)
    X_train, X_val, y_train, y_val = train_test_split(
        X, 
        y, 
        test_size=0.2, 
        random_state=RANDOM_STATE, 
        stratify=y
    )

    print(f"Data loaded successfully.")
    for model in MODELS:
        with mlflow.start_run(run_name=model) as run:
            n_trials = 30
            mlflow.log_param("n_trials", n_trials)

            study = optuna.create_study(direction="maximize")
            study.optimize(
                lambda trial: objective(trial, model, X_train, y_train, X_val, y_val), 
                n_trials=n_trials
            )

            best_trial = study.best_trial
            mlflow.log_params(best_trial.params)

            if best_run_id := best_trial.user_attrs.get("run_id"):
                best_run = mlflow.get_run(best_run_id)

                best_metrics = {
                    f"best_{name}": value
                    for name, value in best_run.data.metrics.items()
                }

                mlflow.log_metrics(best_metrics)
                mlflow.log_param("best_child_run_id", best_run_id)

if __name__ == "__main__":
    print(f"Starting model selection process...")
    main()
