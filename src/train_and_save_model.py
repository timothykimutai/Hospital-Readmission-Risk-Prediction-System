import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.model_training import train_model, save_model

def main():
    # Load data
    print("Loading data...")
    df = pd.read_csv('data/synthetic_readmission_data.csv')
    
    # Prepare features and target
    X = df.drop('readmitted_30_days', axis=1)
    y = df['readmitted_30_days']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Define preprocessing steps
    numeric_features = ['age', 'length_of_stay', 'num_prev_admissions', 
                       'num_procedures', 'num_medications', 'a1c', 
                       'glucose', 'creatinine', 'wbc']
    categorical_features = ['gender', 'race', 'admission_type', 
                          'discharge_disposition']
    binary_features = ['diabetes', 'hypertension', 'copd', 
                      'heart_failure', 'renal_failure']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('binary', 'passthrough', binary_features)
        ])
    
    # Train model
    print("Training model...")
    model = train_model(X_train, y_train, preprocessor, model_type='xgb')
    
    # Save model
    print("Saving model...")
    save_model(model, 'models/readmission_model.pkl')
    print("Model saved successfully!")

if __name__ == "__main__":
    main() 