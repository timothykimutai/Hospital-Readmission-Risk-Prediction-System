import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from src.explainability import explain_model, plot_shap_summary, plot_shap_force_plot
from src.fairness_analysis import calculate_fairness_metrics, plot_fairness_metrics
import numpy as np

# Load model and data
@st.cache_resource
def load_artifacts():
    model = joblib.load('models/readmission_model.pkl')
    df = pd.read_csv('data/synthetic_readmission_data.csv')
    return model, df

model, df = load_artifacts()

# App title
st.title('Hospital Readmission Risk Prediction')

# Sidebar for navigation
st.sidebar.title('Navigation')
options = st.sidebar.radio('Select a page:', 
                          ['Prediction', 'Model Explainability', 'Fairness Analysis'])

if options == 'Prediction':
    st.header('Patient Readmission Risk Prediction')
    
    # Create form for input
    with st.form('patient_form'):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider('Age', 18, 100, 65)
            gender = st.selectbox('Gender', ['Male', 'Female'])
            race = st.selectbox('Race', ['White', 'Black', 'Asian', 'Hispanic', 'Other'])
            admission_type = st.selectbox('Admission Type', ['Emergency', 'Urgent', 'Elective'])
            length_of_stay = st.number_input('Length of Stay (days)', min_value=1, value=5)
            
        with col2:
            discharge_disposition = st.selectbox('Discharge Disposition', 
                                              ['Home', 'SNF', 'HHC', 'AMA', 'Expired', 'Other'])
            num_prev_admissions = st.number_input('Previous Admissions', min_value=0, value=2)
            num_medications = st.number_input('Number of Medications', min_value=0, value=8)
            diabetes = st.checkbox('Diabetes')
            hypertension = st.checkbox('Hypertension')
        
        submitted = st.form_submit_button('Predict Readmission Risk')
    
    if submitted:
        # Create input DataFrame
        input_data = pd.DataFrame({
            'age': [age],
            'gender': [gender],
            'race': [race],
            'admission_type': [admission_type],
            'length_of_stay': [length_of_stay],
            'discharge_disposition': [discharge_disposition],
            'num_prev_admissions': [num_prev_admissions],
            'num_medications': [num_medications],
            'diabetes': [int(diabetes)],
            'hypertension': [int(hypertension)],
            # Add other features with default values
            'num_procedures': [3],
            'copd': [0],
            'heart_failure': [0],
            'renal_failure': [0],
            'a1c': [6.5],
            'glucose': [140],
            'creatinine': [1.2],
            'wbc': [8]
        })
        
        # Make prediction
        proba = model.predict_proba(input_data)[0, 1]
        prediction = model.predict(input_data)[0]
        
        # Display results
        st.subheader('Prediction Results')
        st.metric('Readmission Risk', f"{proba*100:.1f}%")
        
        if prediction == 1:
            st.warning('High risk of readmission within 30 days')
        else:
            st.success('Low risk of readmission within 30 days')
        
        # SHAP explanation
        st.subheader('Prediction Explanation')
        explainer, shap_values, preprocessed_data, feature_names = explain_model(model, input_data)
        st.pyplot(plot_shap_force_plot(explainer, shap_values, preprocessed_data, feature_names))

elif options == 'Model Explainability':
    st.header('Model Explainability')
    
    # SHAP summary plot
    st.subheader('Feature Importance')
    sample_data = df.sample(100, random_state=42)
    explainer, shap_values, preprocessed_data, feature_names = explain_model(model, sample_data)
    st.pyplot(plot_shap_summary(explainer, shap_values, preprocessed_data, feature_names))
    
    # Feature importance table
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('Importance', ascending=False)
    st.dataframe(importance_df)

elif options == 'Fairness Analysis':
    st.header('Fairness Analysis')
    
    # Calculate fairness metrics
    sensitive_attributes = {
        'gender': ['Male', 'Female'],
        'race': ['White', 'Black', 'Asian', 'Hispanic', 'Other']
    }
    
    fairness_results = calculate_fairness_metrics(
        model, 
        df, 
        df['readmitted_30_days'], 
        sensitive_attributes)
    
    # Display results
    for attr, metrics in fairness_results.items():
        st.subheader(f'Fairness Analysis by {attr}')
        
        # Metrics table
        metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
        st.dataframe(metrics_df)
        
        # Plots
        plots = plot_fairness_metrics({attr: metrics})
        st.pyplot(plots[attr])