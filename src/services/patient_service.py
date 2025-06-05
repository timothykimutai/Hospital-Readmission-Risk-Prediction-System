from sqlalchemy.orm import Session
from datetime import date, timedelta
from db.models import Patient, Admission, Prediction
from typing import Optional, List
import json

class PatientService:
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def create_patient(self, patient_data: dict) -> Patient:
        """Create a new patient record"""
        patient = Patient(**patient_data)
        self.db.add(patient)
        self.db.commit()
        self.db.refresh(patient)
        return patient
    
    def create_admission(self, patient_id: int, admission_data: dict) -> Admission:
        """Create a new admission record for a patient"""
        # Calculate length of stay
        if 'admission_date' in admission_data and 'discharge_date' in admission_data:
            delta = admission_data['discharge_date'] - admission_data['admission_date']
            admission_data['length_of_stay'] = delta.days
        
        admission = Admission(patient_id=patient_id, **admission_data)
        self.db.add(admission)
        self.db.commit()
        self.db.refresh(admission)
        return admission
    
    def create_prediction(self, admission_id: int, risk_score: float, 
                         shap_values: dict, threshold: float = 0.5) -> Prediction:
        """Store a prediction result"""
        risk_category = self._categorize_risk(risk_score, threshold)
        
        prediction = Prediction(
            admission_id=admission_id,
            risk_score=risk_score,
            risk_category=risk_category,
            threshold_used=threshold,
            shap_values=json.dumps(shap_values)
        )
        
        self.db.add(prediction)
        self.db.commit()
        self.db.refresh(prediction)
        return prediction
    
    def _categorize_risk(self, risk_score: float, threshold: float) -> str:
        """Categorize risk score into Low/Medium/High"""
        if risk_score < threshold * 0.7:
            return "Low"
        elif risk_score < threshold * 1.3:
            return "Medium"
        else:
            return "High"
    
    def get_patient(self, patient_id: int) -> Optional[Patient]:
        """Retrieve a patient by ID"""
        return self.db.query(Patient).filter(Patient.id == patient_id).first()
    
    def get_admission(self, admission_id: int) -> Optional[Admission]:
        """Retrieve an admission by ID"""
        return self.db.query(Admission).filter(Admission.id == admission_id).first()
    
    def get_patient_admissions(self, patient_id: int) -> List[Admission]:
        """Get all admissions for a patient"""
        return self.db.query(Admission).filter(Admission.patient_id == patient_id).all()
    
    def get_recent_predictions(self, days: int = 30) -> List[Prediction]:
        """Get recent predictions within the last N days"""
        cutoff_date = date.today() - timedelta(days=days)
        return (
            self.db.query(Prediction)
            .filter(Prediction.prediction_date >= cutoff_date)
            .order_by(Prediction.prediction_date.desc())
            .all()
        )
    
    def update_readmission_status(self, admission_id: int, was_readmitted: bool, 
                               readmission_date: Optional[date] = None) -> Admission:
        """Update the readmission status for an admission"""
        admission = self.get_admission(admission_id)
        if admission:
            admission.was_readmitted = was_readmitted
            admission.readmission_date = readmission_date
            self.db.commit()
            self.db.refresh(admission)
        return admission