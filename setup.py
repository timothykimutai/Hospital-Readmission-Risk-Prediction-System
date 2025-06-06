from setuptools import setup, find_packages

setup(
    name="hospital_readmission",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "pandas",
        "joblib",
        "shap",
        "matplotlib",
        "numpy",
        "sqlalchemy",
        "alembic"
    ],
) 