from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, Date, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import datetime

Base = declarative_base()

class Patient(Base):
    __tablename__ = 'patients'
    
    id = Column(Integer, primary_key=True)
    mrn = Column(String(50), unique=True)  # Medical Record Number
    first_name = Column(String(100))
    last_name = Column(String(100))
    date_of_birth = Column(Date)
    gender = Column(String(20))
    race = Column(String(50))
    ethnicity = Column(String(50))
    
    # Admission details
    admissions = relationship("Admission", back_populates="patient")
    
    def age(self):
        return (datetime.date.today() - self.date_of_birth).days // 365

class Admission(Base):
    """Hospital admission record"""
    __tablename__ = 'admissions'
    
    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey('patients.id'))
    admission_date = Column(Date)
    discharge_date = Column(Date)
    admission_type = Column(String(50))  # Emergency, Urgent, Elective
    discharge_disposition = Column(String(100))
    length_of_stay = Column(Integer)
    
    # Medical information
    num_prev_admissions = Column(Integer)
    num_procedures = Column(Integer)
    num_medications = Column(Integer)
    
    # Comorbidities
    diabetes = Column(Boolean)
    hypertension = Column(Boolean)
    copd = Column(Boolean)
    heart_failure = Column(Boolean)
    renal_failure = Column(Boolean)
    
    # Lab results
    a1c = Column(Float)
    glucose = Column(Float)
    creatinine = Column(Float)
    wbc = Column(Float)
    
    # Prediction and outcome
    predictions = relationship("Prediction", back_populates="admission")
    was_readmitted = Column(Boolean)
    readmission_date = Column(Date, nullable=True)
    
    patient = relationship("Patient", back_populates="admissions")

class Prediction(Base):
    """Model prediction results"""
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True)
    admission_id = Column(Integer, ForeignKey('admissions.id'))
    prediction_date = Column(Date, default=datetime.date.today)
    risk_score = Column(Float)  # 0-1 probability
    risk_category = Column(String(20))  # Low, Medium, High
    threshold_used = Column(Float)
    
    # SHAP values (stored as JSON)
    shap_values = Column(String)  
    
    admission = relationship("Admission", back_populates="predictions")