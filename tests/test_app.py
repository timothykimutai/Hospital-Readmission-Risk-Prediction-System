import pytest
from src.app import load_artifacts
import pandas as pd
import numpy as np

def test_load_artifacts():
    """Test that model and data are loaded correctly"""
    model, df = load_artifacts()
    
    # Check that model is loaded
    assert model is not None
    
    # Check that dataframe is loaded and has expected columns
    assert isinstance(df, pd.DataFrame)
    expected_columns = [
        'age', 'gender', 'race', 'admission_type', 'length_of_stay',
        'discharge_disposition', 'num_prev_admissions', 'num_medications',
        'diabetes', 'hypertension', 'heart_failure', 'renal_failure',
        'a1c', 'glucose', 'creatinine', 'wbc', 'num_procedures', 'copd'
    ]
    for col in expected_columns:
        assert col in df.columns

def test_model_prediction():
    """Test that model can make predictions"""
    model, df = load_artifacts()
    
    # Create sample input data
    sample_data = df.iloc[0:1].copy()
    
    # Make prediction
    proba = model.predict_proba(sample_data)[0, 1]
    prediction = model.predict(sample_data)[0]
    
    # Check prediction format
    assert isinstance(proba, (float, np.float64))
    assert 0 <= proba <= 1
    assert prediction in [0, 1] 