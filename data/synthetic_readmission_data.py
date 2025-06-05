import pandas as pd
from faker import Faker
import numpy as np

def generate_synthetic_data(samples= 10000):
    fake = Faker()
    np.random.seed(42)
    
    # Set demographics
    ages = np.random.normal(65,15, samples).astype(int) # Mean (μ) = 65, Standard deviation (σ) = 15
    ages = np.clip(ages, 18, 100) # All values < 18 are set to 18. All values > 100 are set to 100
    genders = np.random.choice(['Male', 'Female'], samples, p=[0.48, 0.52] ) # 48% of the time, the value will be 'Male'. 52% of the time, the value will be 'Female'
    races = np.random.choice(['White', 'Black', 'Asian', 'Hispanic', 'Other'],
                             samples, p=[0.6, 0.2, 0.1, 0.08, 0.02]) # 60% White, 20% Black, 10% Asian, 8% Hispanic, 2% Other
    
    # Admission details
    admission_types = np.random.choice(['Emergency', 'Urgent', 'Elective'], 
                                     samples, p=[0.7, 0.2, 0.1])
    lengths_of_stay = np.random.lognormal(1.5, 0.7, samples).astype(int)
    discharge_dispositions = np.random.choice(
         ['Home', 'SNF', 'HHC', 'AMA', 'Expired', 'Other'],
         samples, p=[0.6, 0.15, 0.15, 0.02, 0.03, 0.05]
    )
    # Medical History
    num_prev_admissions = np.random.poisson(2, samples) # Poisson distribution with a mean (λ) = 2
    num_procedures = np.random.poisson(3, samples) # Mean (λ) = 3
    num_medications = np.random.poisson(8, samples) # Mean (λ) = 8 
    
    # Comorbidities (as binary flags)
    comorbidities = ['diabetes', 'hypertension', 'copd', 'heart_failure', 'renal_failure'] # list of common chronic conditions (comorbidities)
    comorbidity_data = {
         comorbidity: np.random.binomial(1, np.random.uniform(0.1, 0.7)) # Drawing one trial.Random probability between 0.1 and 0.7
         for comorbidity in comorbidities
    }
    # Lab Results
    lab_data = {
        'a1c': np.random.normal(6.5, 2, samples), # Hemoglobin A1C (blood sugar avg.)
        'glucose': np.random.normal(140, 40, samples), # Blood glucose level
        'creatinine': np.random.normal(1.2, 0.5, samples), # Kidney function indicator
        'wbc': np.random.normal(8, 3, samples) # White blood cell count
    }
    
     # Target variable - readmission within 30 days
    readmit_probs = 0.1 + 0.3*(ages > 70) + 0.2*(num_prev_admissions > 3) + \
                    0.15*(comorbidity_data['heart_failure'] == 1) + \
                    0.1*(lengths_of_stay > 7) - 0.1*(discharge_dispositions == 'Home')
    readmit_probs = np.clip(readmit_probs, 0.05, 0.8)
    readmitted = np.random.binomial(1, readmit_probs)
    
    # Combined data
    data = {
        'age': ages,
        'gender': genders,
        'race': races,
        'admission_type': admission_types,
        'length_of_stay': lengths_of_stay,
        'discharge_disposition': discharge_dispositions,
        'num_prev_admissions': num_prev_admissions,
        'num_procedures': num_procedures,
        'num_medications': num_medications,
        **comorbidity_data,
        **lab_data,
        'readmitted_30_days': readmitted
    }
    df = pd.DataFrame(data)
    return df
if __name__ == "__main__":
    df = generate_synthetic_data(10000)
    df.to_csv('data/synthetic_readmission_data.csv', index=False)