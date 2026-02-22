import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit
from sklearn.metrics import precision_score, recall_score, f1_score
import json

script_dir = Path(__file__).parent  # Script directory
project_root = script_dir.parent     # Parent folder (project root)
data_path = project_root / 'processed_data/features.parquet'

# Read the data and split X, y
data = pd.read_parquet(data_path)
X = data.drop(columns=['genre'])
y = data['genre']

# Make train-validation-test split (70% train, 15% validation, 15% test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

# Hyperparameter search space for SVM
grid_svm = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__gamma': ['scale', 0.01, 0.1, 1],
    'classifier__kernel': ['rbf'],
    'classifier__decision_function_shape': ['ovr', 'ovo']
}

# Hyperparameter search space for Random Forest
grid_rf = {
    'classifier__n_estimators': [100, 150, 200],
    'classifier__max_depth': [None, 10, 15, 20],
    'classifier__min_samples_split': [2, 4, 6],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_features': ['sqrt', 'log2', None]
}

# Hyperparameter search space for XGBoost
grid_xgb = {
    'classifier__learning_rate': [0.01, 0.1, 1.0],
    'classifier__max_depth': [3,6,10],
    'classifier__objective': ['multi:softmax', 'multi:softprob']
}

models = [
    ('SVM', SVC(random_state=42), grid_svm),
    ('Random Forest', RandomForestClassifier(random_state=42), grid_rf),
    ('XGBoost', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'), grid_xgb)
]

# Combine train and validation sets for PredefinedSplit
X_combined = np.vstack((X_train, X_val))
y_combined = np.hstack((y_train, y_val))

split_index = [-1] * len(X_train) + [0] * len(X_val)
pds = PredefinedSplit(test_fold=split_index)

# List to store results
results_list = []

# Train each model and build results inside the loop
for name, model, grid in models:
    if name == 'SVM':
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
    else:
        pipeline = Pipeline([
            ('classifier', model)
        ])
    
    print(f"Training {name}...")
    cv = GridSearchCV(pipeline, grid, cv=pds, scoring='f1_weighted', n_jobs=-1)
    cv.fit(X_combined, y_combined)

    # Evaluate on validation set
    val_score = cv.score(X_val, y_val)
    y_val_pred = cv.best_estimator_.predict(X_val)
    val_precision = precision_score(y_val, y_val_pred, average='weighted')
    val_recall = recall_score(y_val, y_val_pred, average='weighted')
    val_f1 = f1_score(y_val, y_val_pred, average='weighted')

    # Evaluate on test set
    test_score = cv.score(X_test, y_test)
    y_test_pred = cv.best_estimator_.predict(X_test)
    test_precision = precision_score(y_test, y_test_pred, average='weighted')
    test_recall = recall_score(y_test, y_test_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')

    # Add results to list
    results_list.append({
        'Model': name,
        'CV_Score': round(cv.best_score_, 4),
        'Val_Score': round(val_score, 4),
        'Val_Precision': round(val_precision, 4),
        'Val_Recall': round(val_recall, 4),
        'Val_F1': round(val_f1, 4),
        'Test_Score': round(test_score, 4),
        'Test_Precision': round(test_precision, 4),
        'Test_Recall': round(test_recall, 4),
        'Test_F1': round(test_f1, 4),
        'Best_Params': str(cv.best_params_)
    })

    print(f"{name} - CV: {cv.best_score_:.4f}, Val: {val_score:.4f}, Test: {test_score:.4f}")

# Create DataFrame from results
results_df = pd.DataFrame(results_list)

# Save results to CSV
results_csv_path = project_root / 'results/model_results.csv'
results_csv_path.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(results_csv_path, index=False)

print(f"\nResults saved to: {results_csv_path}")
print(results_df)