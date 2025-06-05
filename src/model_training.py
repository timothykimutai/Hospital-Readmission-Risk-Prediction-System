from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (roc_auc_score, precision_score, 
                            recall_score, f1_score, confusion_matrix)
from sklearn.pipeline import Pipeline
import joblib
import pandas as pd

def train_model(X_train, y_train, preprocessor, model_type='xgb'):
    # Full pipeline
    if model_type == 'xgb':
        model = XGBClassifier(objective='binary:logistic', 
                            eval_metric='logloss',
                            random_state=42,
                            use_label_encoder=False)
        
        param_grid = {
            'model__n_estimators': [100, 200],
            'model__max_depth': [3, 6],
            'model__learning_rate': [0.01, 0.1],
            'model__subsample': [0.8, 1.0],
            'model__colsample_bytree': [0.8, 1.0]
        }
    else:
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'model__n_estimators': [100, 200],
            'model__max_depth': [None, 10, 20],
            'model__min_samples_split': [2, 5],
            'model__min_samples_leaf': [1, 2]
        }
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)])
    # Grid search with 3-fold CV
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, 
                             scoring='roc_auc', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV AUC: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    print("Model Performance:")
    for name, value in metrics.items():
        print(f"{name:10}: {value:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    return metrics
def save_model(model, filepath):
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")
def load_model(filepath):
    return joblib.load(filepath)

def calculate_average_odds_difference(y_true, y_pred, sensitive_attribute):
    """
    Calculate the Average Odds Difference (AOD) fairness metric.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        sensitive_attribute: Binary sensitive attribute (e.g., gender)
    
    Returns:
        float: Average Odds Difference
    """
    # Calculate confusion matrices for each group
    group_0_mask = sensitive_attribute == 0
    group_1_mask = sensitive_attribute == 1
    
    cm_0 = confusion_matrix(y_true[group_0_mask], y_pred[group_0_mask])
    cm_1 = confusion_matrix(y_true[group_1_mask], y_pred[group_1_mask])
    
    # Calculate FPR and TPR for each group
    def get_rates(cm):
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        return fpr, tpr
    
    fpr_0, tpr_0 = get_rates(cm_0)
    fpr_1, tpr_1 = get_rates(cm_1)
    
    # Calculate Average Odds Difference
    avg_odds_diff = 0.5 * (
        (fpr_0 - fpr_1) +  # Difference in false positive rates
        (tpr_0 - tpr_1)    # Difference in true positive rates
    )
    
    return avg_odds_diff