import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.database import db
from db.models import Patient, Admission, Prediction
from src.services.patient_service import PatientService
from datetime import date, timedelta
import pandas as pd

def test_database_connection():
    # Test basic database connectivity
    try:
        # Initialize database
        db.init_db()
        session = db.get_session()
        
        # Test creating a patient
        patient_service = PatientService(session)
        
        # Create test patient
        patient_data = {
            "first_name": "Test",
            "last_name": "Patient",
            "date_of_birth": date.today() - timedelta(days=30*365),  # 30 years old
            "gender": "Male",
            "race": "White",
            "ethnicity": "Non-Hispanic"
        }
        
        patient = patient_service.create_patient(patient_data)
        print("✅ Successfully created patient:", patient.id)
        
        # Test creating an admission
        admission_data = {
            "admission_date": date.today() - timedelta(days=5),
            "discharge_date": date.today(),
            "admission_type": "Emergency",
            "discharge_disposition": "Home",
            "num_prev_admissions": 1,
            "num_medications": 5,
            "diabetes": True,
            "hypertension": False,
            "heart_failure": False,
            "renal_failure": False,
            "a1c": 6.5,
            "glucose": 140,
            "creatinine": 1.2,
            "wbc": 8
        }
        
        admission = patient_service.create_admission(patient.id, admission_data)
        print("✅ Successfully created admission:", admission.id)
        
        # Test creating a prediction
        prediction_data = {
            "expected_value": 0.5,
            "shap_values": [[0.1, 0.2, 0.3]],
            "features": [[1, 2, 3]],
            "feature_names": ["feature1", "feature2", "feature3"]
        }
        
        prediction = patient_service.create_prediction(admission.id, 0.75, prediction_data)
        print("✅ Successfully created prediction:", prediction.id)
        
        # Test retrieving data
        retrieved_patient = patient_service.get_patient(patient.id)
        print("✅ Successfully retrieved patient:", retrieved_patient.first_name, retrieved_patient.last_name)
        
        retrieved_admission = patient_service.get_admission(admission.id)
        print("✅ Successfully retrieved admission:", retrieved_admission.admission_type)
        
        # Test retrieving patient admissions
        admissions = patient_service.get_patient_admissions(patient.id)
        print("✅ Successfully retrieved patient admissions:", len(admissions))
        
        # Test retrieving recent predictions
        predictions = patient_service.get_recent_predictions(days=30)
        print("✅ Successfully retrieved recent predictions:", len(predictions))
        
        print("\n🎉 All database tests passed successfully!")
        
    except Exception as e:
        print("❌ Database test failed:", str(e))
        raise e
    finally:
        session.close()

if __name__ == "__main__":
    test_database_connection() 