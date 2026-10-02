import json
import pickle
import time
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import mlflow

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sqlalchemy import null
from xgboost import XGBClassifier

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("model_selection")

RANDOM_STATE = 42
TARGET_COLUMN = "genre"
MODELS = ["SVC", "RandomForest", "XGBClassifier"]

# MODEL_GRIDS = {
#   "SVM": {
#     "classifier__C": [0.1, 1, 10, 100],
#     "classifier__gamma": ["scale", 0.01, 0.1, 1],
#     "classifier__kernel": ["rbf"],
#     "classifier__decision_function_shape": ["ovr", "ovo"]
#   },
#   "Random Forest": {
#     "classifier__n_estimators": [100, 150, 200],
#     "classifier__max_depth": [null, 10, 15, 20],
#     "classifier__min_samples_split": [2, 4, 6],
#     "classifier__min_samples_leaf": [1, 2, 4],
#     "classifier__max_features": ["sqrt", "log2", null]
#   },
#   "XGBoost": {
#     "classifier__learning_rate": [0.01, 0.1, 1.0],
#     "classifier__max_depth": [3, 6, 10],
#     "classifier__objective": ["multi:softmax", "multi:softprob"]
#   }
# }

def load_features(data_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load features parquet and split into predictors and target."""
    data = pd.read_parquet(data_path)
    features = data.drop(columns=[TARGET_COLUMN])
    target = data[TARGET_COLUMN]
    return features, target


def split_dataset(features: pd.DataFrame, target: pd.Series):
    """Create train, validation and test sets with a 70/15/15 split."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        features,
        target,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def load_genre_encoder(encoder_path: Path):
    """Load genre encoder (LabelEncoder) from pickle file."""
    with encoder_path.open("rb") as file:
        encoder = pickle.load(file)
    return encoder

def train_models(X_train, y_train, X_test, y_test, MODEL_GRIDS = MODELS, ):
    models = {}
    """Train models using the provided hyperparameter grids."""
    for model_name in MODEL_GRIDS:
        if model_name == "SVC":
            model = make_pipeline(StandardScaler(), SVC(random_state=RANDOM_STATE, probability=True))
        elif model_name == "RandomForest":
            model = RandomForestClassifier(random_state=RANDOM_STATE)
        elif model_name == "XGBClassifier":
            model = XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='mlogloss')
        else:
            raise ValueError(f"Unsupported model: {model_name}")


        print(f"Training {model_name}...")
        start_time = time.perf_counter()
        if model_name == "SVC":
            model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)
        models[model_name] = model

        print(f"{model_name} training completed in {time.perf_counter() - start_time:.2f} seconds.")

        with mlflow.start_run(run_name=model_name):
            # Log model parameters
            mlflow.log_params(model.get_params())

            # Make predictions on the test set
            y_pred = model.predict(X_test)

            # Calculate evaluation metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            # Log evaluation metrics
            mlflow.log_metrics({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            })

        print(f"{model_name} loaded in MLflow")

    return models
                             

def main() -> None:
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    data_path = project_root / "processed_data/features.parquet"
    grids_path = project_root / "configs/model_grids.json"
    encoder_path = project_root / "model/genre_encoder.pkl"
    results_path = project_root / "model/model_selection/model_results.csv"
    confusion_matrices_dir = project_root / "model/model_selection/confusion_matrices"
    models_output_dir = project_root / "model/model_selection"

    genre_encoder = load_genre_encoder(encoder_path)
    X, y = load_features(data_path)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

    print(f"Data loaded successfully.")
    models = train_models(X_train, y_train, X_test, y_test)



if __name__ == "__main__":
    print(f"Starting model selection process...")
    main()
