import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import json

from src.explainability import explain_model, plot_shap_summary, plot_shap_force_plot
from src.fairness_analysis import calculate_fairness_metrics, plot_fairness_metrics
from db.database import db
from db.models import Patient, Admission, Prediction
from src.services.patient_service import PatientService

# Initialize database and services
db.init_db()
patient_service = PatientService(db.get_session())

# Page configuration
st.set_page_config(
    page_title="Hospital Readmission Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and data
@st.cache_resource
def load_artifacts():
    model = joblib.load('models/readmission_model.pkl')
    df = pd.read_csv('data/synthetic_readmission_data.csv')
    return model, df

model, df = load_artifacts()

# Custom CSS for styling
st.markdown("""
    <style>
    /* Background image */
    .stApp {
        background-image: url("https://pixabay.com/photos/hospital-ward-hospital-1338585/");
        background-attachment: fixed;
        background-size: cover;
        background-position: center;
    }
    
    /* Main container styling */
    .main {
        background-color: rgba(0, 0, 0, 0.7);
        color: white;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #ffffff !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }
    
    /* Card styling */
    .stForm {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 2rem;
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 8px;
        padding: 0.5em 1em;
        border: none;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .stButton > button:hover {
        background-color: #1668a1;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: rgba(0, 0, 0, 0.8);
    }
    
    /* Form elements styling */
    .stSelectbox, .stSlider, .stNumberInput {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 5px;
        padding: 0.5rem;
    }
    
    /* Metric styling */
    .stMetric {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        backdrop-filter: blur(5px);
    }
    
    /* Dataframe styling */
    .stDataFrame {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Success and warning messages */
    .stSuccess, .stWarning {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.title("🏥 Readmission Risk App")
    st.markdown("""---""")
    selected_tab = st.radio("Navigate", ["Prediction", "Model Explainability", "Fairness Analysis"])
    st.markdown("""---""")
    st.markdown("Built with ❤️ using Streamlit")

# Prediction Tab
if selected_tab == "Prediction":
    st.title("🔍 Patient Readmission Risk Prediction")
    with st.container():
        with st.form("prediction_form"):
            st.markdown("### 📋 Patient Information")
            col1, col2 = st.columns(2)
            with col1:
                first_name = st.text_input("First Name")
                last_name = st.text_input("Last Name")
                age = st.slider("Age", 18, 100, 65)
                gender = st.selectbox("Gender", ["Male", "Female"])
                race = st.selectbox("Race", ["White", "Black", "Asian", "Hispanic", "Other"])
                admission_type = st.selectbox("Admission Type", ["Emergency", "Urgent", "Elective"])
                length_of_stay = st.number_input("Length of Stay (days)", min_value=1, value=5)
            
            with col2:
                discharge_disposition = st.selectbox("Discharge Disposition", ["Home", "SNF", "HHC", "AMA", "Expired", "Other"])
                num_prev_admissions = st.number_input("Previous Admissions", min_value=0, value=2)
                num_medications = st.number_input("Number of Medications", min_value=0, value=8)
                diabetes = st.checkbox("Diabetes")
                hypertension = st.checkbox("Hypertension")
                heart_failure = st.checkbox("Heart Failure")
                renal_failure = st.checkbox("Renal Failure")
                a1c_value = st.number_input("A1C", min_value=4.0, max_value=15.0, value=6.5, step=0.1)
                glucose_value = st.number_input("Glucose", min_value=50, max_value=500, value=140)
                creatinine_value = st.number_input("Creatinine", min_value=0.1, max_value=10.0, value=1.2, step=0.1)
                wbc_value = st.number_input("WBC", min_value=1, max_value=50, value=8)

            submitted = st.form_submit_button("Predict Readmission Risk")
    
    if submitted:
        # Create patient record
        patient_data = {
            "first_name": first_name,
            "last_name": last_name,
            "date_of_birth": datetime.now().date() - timedelta(days=age*365),
            "gender": gender,
            "race": race,
            "ethnicity": "Unknown"
        }
        
        patient = patient_service.create_patient(patient_data)
        
        # Create admission record
        admission_data = {
            "admission_date": datetime.now().date() - timedelta(days=length_of_stay),
            "discharge_date": datetime.now().date(),
            "admission_type": admission_type,
            "discharge_disposition": discharge_disposition,
            "num_prev_admissions": num_prev_admissions,
            "num_medications": num_medications,
            "diabetes": diabetes,
            "hypertension": hypertension,
            "heart_failure": heart_failure,
            "renal_failure": renal_failure,
            "a1c": a1c_value,
            "glucose": glucose_value,
            "creatinine": creatinine_value,
            "wbc": wbc_value,
            "num_procedures": 3,
            "copd": 0
        }
        
        admission = patient_service.create_admission(patient.id, admission_data)
        
        # Prepare input data for prediction
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
            'heart_failure': [int(heart_failure)],
            'renal_failure': [int(renal_failure)],
            'a1c': [a1c_value],
            'glucose': [glucose_value],
            'creatinine': [creatinine_value],
            'wbc': [wbc_value],
            'num_procedures': [3],
            'copd': [0]
        })
        
        # Make prediction
        proba = model.predict_proba(input_data)[0, 1]
        prediction = model.predict(input_data)[0]
        
        # Get SHAP values
        explainer, shap_values, preprocessed_data, feature_names = explain_model(model, input_data)
        shap_dict = {
            "expected_value": float(explainer.expected_value),
            "shap_values": shap_values.tolist(),
            "features": preprocessed_data.tolist(),
            "feature_names": feature_names
        }
        
        # Store prediction
        prediction_record = patient_service.create_prediction(
            admission.id, proba, shap_dict)

        # Display results
        st.markdown("### 📊 Prediction Result")
        st.metric("Readmission Risk", f"{proba * 100:.1f}%")
        
        if prediction == 1:
            st.warning("⚠️ High risk of readmission within 30 days")
        else:
            st.success("✅ Low risk of readmission within 30 days")
        
        st.markdown("### 🧠 SHAP Explanation")
        st.pyplot(plot_shap_force_plot(explainer, shap_values, preprocessed_data, feature_names))

# Explainability Tab
elif selected_tab == "Model Explainability":
    st.title("📈 Model Explainability")
    st.markdown("### 🔬 SHAP Summary Plot")
    sample_data = df.sample(100, random_state=42)
    explainer, shap_values, preprocessed_data, feature_names = explain_model(model, sample_data)
    st.pyplot(plot_shap_summary(explainer, shap_values, preprocessed_data, feature_names))

    st.markdown("### 📋 Feature Importance Table")
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('Importance', ascending=False)
    st.dataframe(importance_df)

# Fairness Tab
elif selected_tab == "Fairness Analysis":
    st.title("⚖️ Fairness Analysis")
    
    sensitive_attributes = {
        'gender': ['Male', 'Female'],
        'race': ['White', 'Black', 'Asian', 'Hispanic', 'Other']
    }
    
    fairness_results = calculate_fairness_metrics(
        model, 
        df, 
        df['readmitted_30_days'], 
        sensitive_attributes
    )
    
    for attr, metrics in fairness_results.items():
        st.markdown(f"### 🔍 Fairness by {attr.title()}")
        metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
        st.dataframe(metrics_df)
        plots = plot_fairness_metrics({attr: metrics})
        st.pyplot(plots[attr])