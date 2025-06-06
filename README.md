# Hospital Readmission Risk Prediction System

A machine learning-based web application that predicts the risk of hospital readmission for patients within 30 days of discharge. The system uses patient data, medical history, and clinical measurements to make predictions and provides explanations for its decisions.

## Features

- 🔍 **Risk Prediction**: Predicts the likelihood of patient readmission within 30 days
- 📊 **Model Explainability**: Provides SHAP (SHapley Additive exPlanations) visualizations to understand model decisions
- ⚖️ **Fairness Analysis**: Analyzes model performance across different demographic groups
- 💾 **Data Storage**: Stores patient records, admissions, and predictions in a database
- 🎨 **Modern UI**: Clean and intuitive user interface with responsive design
- 🔄 **CI/CD Pipeline**: Automated testing, linting, and deployment

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Machine Learning**: scikit-learn, SHAP
- **Database**: SQLAlchemy
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib
- **CI/CD**: GitHub Actions
- **Testing**: pytest, flake8, black

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/timothykimutai/Hospital-Readmission-Risk-Prediction-System.git
cd Hospital-Readmission-Risk-Prediction-System
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Initialize the database:
```bash
python -c "from db.database import db; db.init_db()"
```

5. Run the application:
```bash
streamlit run src/app.py
```

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black .
flake8 .
```

### CI/CD Pipeline

The project uses GitHub Actions for continuous integration and deployment:

1. **Testing**: Runs on every push and pull request
   - Unit tests with pytest
   - Code coverage reporting
   - Linting with flake8
   - Code formatting check with black

2. **Deployment**: Runs on main branch after successful tests
   - Deploys to Streamlit Cloud
   - Requires STREAMLIT_API_KEY secret

To set up the CI/CD pipeline:
1. Fork the repository
2. Add your STREAMLIT_API_KEY to repository secrets
3. Enable GitHub Actions in your repository settings

## Project Structure

```
hospital-readmission-prediction/
├── .github/
│   └── workflows/
│       └── test.yml
├── data/
│   └── synthetic_readmission_data.csv
├── models/
│   └── readmission_model.pkl
├── src/
│   ├── app.py
│   ├── explainability.py
│   ├── fairness_analysis.py
│   └── services/
│       └── patient_service.py
├── tests/
│   └── test_app.py
├── db/
│   ├── database.py
│   └── models.py
├── requirements.txt
└── README.md
```

## Usage

1. **Prediction Tab**
   - Enter patient information
   - Fill in medical history and clinical measurements
   - Click "Predict Readmission Risk" to get the prediction
   - View the risk percentage and SHAP explanation

2. **Model Explainability Tab**
   - View SHAP summary plot
   - Explore feature importance table

3. **Fairness Analysis Tab**
   - Analyze model performance across different demographic groups
   - View fairness metrics and visualizations

## Model Details

The system uses a machine learning model trained on historical patient data to predict readmission risk. The model considers various factors including:

- Patient demographics
- Medical history
- Clinical measurements
- Hospital stay information
- Previous admissions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset: Synthetic hospital readmission data
- SHAP library for model explainability
- Streamlit for the web interface
