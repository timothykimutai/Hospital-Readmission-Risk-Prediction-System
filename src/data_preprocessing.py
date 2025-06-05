import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df
def preprocess_data(df):
    # Feature Engineering
    df["age_group"] = pd.cut(df['age'],
                             bins=[18, 30, 50, 65, 75, 100],
                             labels=['18-30', '31-50', '51-65', '66-75', '75+'])
    # Features and target definition
    X = df.drop('readmitted_30_days', axis=1)
    y = df['readmitted_30_days']
    # Define categorical and numeric features
    categorical_features = ['gender', 'race', 'admission_type', 
                          'discharge_disposition', 'age_group']
    numeric_features = ['age', 'length_of_stay', 'num_prev_admissions',
                      'num_procedures', 'num_medications', 'a1c',
                      'glucose', 'creatinine', 'wbc']
    binary_features = ['diabetes', 'hypertension', 'copd', 
                     'heart_failure', 'renal_failure']
    # Preprocessing Pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('binary', 'passthrough', binary_features)])
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test, preprocessor
def get_feature_names(preprocessor, X):
    # Numeric features
    numeric_features = preprocessor.named_transformers_['num'].named_steps['scaler'].get_feature_names_out()
    # Categorical features
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    categorical_features = cat_encoder.get_feature_names_out(
        preprocessor.named_transformers_['cat'].features)
    # Binary features
    binary_features = preprocessor.named_transformers_['binary'].features
    # Combine all
    all_features = np.concatenate([numeric_features, categorical_features, binary_features])
    
    return all_features