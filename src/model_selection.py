import json
import pickle
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

RANDOM_STATE = 42
TARGET_COLUMN = "genre"
REQUIRED_MODELS = ("SVM", "Random Forest", "XGBoost")


def load_model_grids(config_path: Path) -> dict:
    """Load hyperparameter grids from JSON and validate required model keys."""
    with config_path.open("r", encoding="utf-8") as file:
        grids = json.load(file)

    missing = [model_name for model_name in REQUIRED_MODELS if model_name not in grids]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Missing model grids in JSON: {missing_text}")

    return grids


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


def build_model_registry(model_grids: dict) -> list[tuple[str, object, dict, bool]]:
    """Map each model name to its estimator, grid, and whether scaling is required."""
    return [
        ("SVM", SVC(random_state=RANDOM_STATE), model_grids["SVM"], True),
        (
            "Random Forest",
            RandomForestClassifier(random_state=RANDOM_STATE),
            model_grids["Random Forest"],
            False,
        ),
        (
            "XGBoost",
            XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="mlogloss",
            ),
            model_grids["XGBoost"],
            False,
        ),
    ]


def make_pipeline(estimator, with_scaler: bool) -> Pipeline:
    """Build a pipeline and include feature scaling only when needed."""
    steps = []
    if with_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", estimator))
    return Pipeline(steps)


def load_genre_encoder(encoder_path: Path):
    """Load genre encoder (LabelEncoder) from pickle file."""
    with encoder_path.open("rb") as file:
        encoder = pickle.load(file)
    return encoder


def compute_metrics(y_true, y_pred, average="macro") -> tuple[float, float, float]:
    """Return precision, recall and F1 metrics."""
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    return precision, recall, f1


def save_trained_model(model, model_name: str, output_dir: Path) -> None:
    """Save trained model pipeline using joblib."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{model_name.lower().replace(' ', '_')}_model.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved: {model_path}")


def plot_confusion_matrix(y_true, y_pred, model_name: str, genre_encoder, output_dir: Path) -> None:
    """Generate and save confusion matrix heatmap for a model."""
    cm = confusion_matrix(y_true, y_pred)
    genre_labels = genre_encoder.classes_

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=genre_labels,
        yticklabels=genre_labels,
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("True Genre")
    plt.xlabel("Predicted Genre")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved: {output_path}")


def train_and_score_models(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    model_registry: list[tuple[str, object, dict, bool]],
    genre_encoder,
    confusion_matrices_dir: Path,
    models_output_dir: Path,
) -> pd.DataFrame:
    """Run grid search on training set only, evaluate on validation and test sets.
    
    Process:
    1. GridSearchCV finds best hyperparams using 5-fold CV within train set (Tuning_CV).
    2. Evaluate best model on validation set (clean, Val_Score).
    3. Refit on train+val and evaluate on test set (final assessment, Test_Score).
    4. Generate confusion matrix heatmap for test predictions.
    5. Save trained model for later inference.
    """
    results = []

    for model_name, estimator, grid, with_scaler in model_registry:
        print(f"Training {model_name}...")
        pipeline = make_pipeline(estimator, with_scaler)

        # Grid search on training set only (no contamination from val).
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=grid,
            cv=5,
            scoring="f1_macro",
            refit=True,
        )
        search.fit(X_train, y_train)

        # Evaluate on validation set (clean, no data leakage).
        val_score = search.score(X_val, y_val)
        y_val_pred = search.best_estimator_.predict(X_val)
        val_precision, val_recall, val_f1 = compute_metrics(y_val, y_val_pred, average="macro")

        # Refit best model on train+val for test evaluation.
        X_combined = pd.concat([X_train, X_val], axis=0)
        y_combined = pd.concat([y_train, y_val], axis=0)
        search.best_estimator_.fit(X_combined, y_combined)

        # Evaluate on test set (final assessment).
        test_score = search.score(X_test, y_test)
        start_time = time.perf_counter()
        y_test_pred = search.best_estimator_.predict(X_test)
        elapsed = time.perf_counter() - start_time
        avg_inference_ms = (elapsed / max(len(X_test), 1)) * 1000
        test_precision, test_recall, test_f1_macro = compute_metrics(
            y_test,
            y_test_pred,
            average="macro",
        )
        test_f1_weighted = f1_score(y_test, y_test_pred, average="weighted", zero_division=0)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        # Generate and save confusion matrix.
        plot_confusion_matrix(y_test, y_test_pred, model_name, genre_encoder, confusion_matrices_dir)

        # Save trained model for inference.
        save_trained_model(search.best_estimator_, model_name, models_output_dir)

        results.append(
            {
                "Model": model_name,
                "Tuning_CV_F1": round(search.best_score_, 4),
                "Test_F1_Macro": round(test_f1_macro, 4),
                "Test_Accuracy": round(test_accuracy, 4),
                "Test_F1_Weighted": round(test_f1_weighted, 4),
                "Test_Precision_Macro": round(test_precision, 4),
                "Test_Recall_Macro": round(test_recall, 4),
                "Test_Inference_ms": round(avg_inference_ms, 4),
            }
        )

        print(
            f"{model_name} - Tuning_CV: {search.best_score_:.4f}, "
            f"Val: {val_score:.4f}, Test: {test_score:.4f}"
        )

    return pd.DataFrame(results)


def save_results(results_df: pd.DataFrame, results_path: Path) -> None:
    """Save model comparison results to CSV."""
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    print(results_df)


def main() -> None:
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    data_path = project_root / "processed_data/features.parquet"
    grids_path = project_root / "configs/model_grids.json"
    encoder_path = project_root / "model/genre_encoder.pkl"
    results_path = project_root / "model/model_selection/model_results.csv"
    confusion_matrices_dir = project_root / "model/model_selection/confusion_matrices"
    models_output_dir = project_root / "model/model_selection"

    model_grids = load_model_grids(grids_path)
    genre_encoder = load_genre_encoder(encoder_path)
    X, y = load_features(data_path)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
    model_registry = build_model_registry(model_grids)

    results_df = train_and_score_models(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        model_registry,
        genre_encoder,
        confusion_matrices_dir,
        models_output_dir,
    )
    # Save X train, y train, X val, y val, X test, y test in parquet format
    split_output_dir = project_root / "processed_data/train_val_test"
    split_output_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_parquet(split_output_dir / "X_train.parquet")
    y_train.to_frame(name=TARGET_COLUMN).to_parquet(split_output_dir / "y_train.parquet")
    X_val.to_parquet(split_output_dir / "X_val.parquet")
    y_val.to_frame(name=TARGET_COLUMN).to_parquet(split_output_dir / "y_val.parquet")
    X_test.to_parquet(split_output_dir / "X_test.parquet")
    y_test.to_frame(name=TARGET_COLUMN).to_parquet(split_output_dir / "y_test.parquet")

    save_results(results_df, results_path)



if __name__ == "__main__":
    main()