import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def explain_model(model, data, feature_names=None):
    """
    Generate SHAP explanations for the model predictions.
    
    Args:
        model: Trained model pipeline
        data: Input data (DataFrame)
        feature_names: List of feature names (optional)
    
    Returns:
        tuple: (explainer, shap_values, preprocessed_data)
    """
    # Get the preprocessor from the pipeline
    preprocessor = model.named_steps['preprocessor']
    
    # Transform the data
    preprocessed_data = preprocessor.transform(data)
    
    # Get feature names after preprocessing
    if feature_names is None:
        feature_names = get_feature_names(preprocessor, data.columns)
    
    # Create explainer
    explainer = shap.TreeExplainer(model.named_steps['model'])
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(preprocessed_data)
    
    # If binary classification, take the second class's SHAP values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    return explainer, shap_values, preprocessed_data, feature_names

def get_feature_names(preprocessor, original_features):
    """
    Get feature names after preprocessing.
    
    Args:
        preprocessor: ColumnTransformer
        original_features: Original feature names
    
    Returns:
        list: Transformed feature names
    """
    feature_names = []
    
    for name, trans, cols in preprocessor.transformers_:
        if name == 'num':
            feature_names.extend(cols)
        elif name == 'cat':
            # Get categories for each categorical feature
            encoder = trans
            for i, col in enumerate(cols):
                categories = encoder.categories_[i]
                feature_names.extend([f"{col}_{cat}" for cat in categories])
        elif name == 'binary':
            feature_names.extend(cols)
    
    return feature_names

def plot_shap_summary(explainer, shap_values, preprocessed_data, feature_names):
    """
    Plot SHAP summary plot.
    
    Args:
        explainer: SHAP explainer
        shap_values: SHAP values
        preprocessed_data: Preprocessed input data
        feature_names: List of feature names
    
    Returns:
        matplotlib figure
    """
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        preprocessed_data,
        feature_names=feature_names,
        show=False
    )
    return plt.gcf()

def plot_shap_force_plot(explainer, shap_values, preprocessed_data, feature_names):
    """
    Plot SHAP force plot for a single prediction.
    
    Args:
        explainer: SHAP explainer
        shap_values: SHAP values
        preprocessed_data: Preprocessed input data
        feature_names: List of feature names
    
    Returns:
        matplotlib figure
    """
    plt.figure(figsize=(10, 3))
    shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        preprocessed_data[0],
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )
    return plt.gcf()

def get_feature_importance(shap_values, feature_names):
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)
    
    return importance_df