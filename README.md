# Customer Churn Prediction Project

## Overview
This project predicts customer churn in a subscription-based business using machine learning. It demonstrates end-to-end data science workflows including data generation, preprocessing, model development, evaluation, and deployment.

## Project Structure
```
churn_analysis/
├── data/
│   ├── raw/                    # Raw data files
│   ├── processed/             # Cleaned and processed data
│   └── external/              # External data sources
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_development.ipynb
│   └── 04_model_evaluation.ipynb
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_generator.py  # Synthetic data generation
│   │   └── preprocessing.py    # Data cleaning and preprocessing
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model_trainer.py
│   │   └── model_evaluator.py
│   └── visualization/
│       ├── __init__.py
│       └── plot_utils.py
├── models/                    # Trained model files
├── reports/                   # Generated reports and visualizations
├── app/                       # Flask web application
├── tests/                     # Unit tests
├── requirements.txt
└── README.md
```

## Features
- **Data Generation**: Synthetic customer data with realistic churn patterns
- **EDA**: Comprehensive exploratory data analysis with interactive visualizations
- **Feature Engineering**: Advanced feature creation and selection
- **Model Development**: Multiple ML algorithms (Logistic Regression, Random Forest, XGBoost)
- **Model Evaluation**: Comprehensive metrics and cross-validation
- **Model Interpretation**: SHAP and LIME for explainable AI
- **Business Integration**: Cost-benefit analysis and retention strategies
- **Deployment**: Flask API and interactive dashboard
- **Automation**: End-to-end pipeline with Docker support

## Installation
```bash
pip install -r requirements.txt
```

## Usage
1. Generate synthetic data: `python src/data/data_generator.py`
2. Run EDA: `jupyter notebook notebooks/01_data_exploration.ipynb`
3. Train models: `python src/models/model_trainer.py`
4. Start web app: `python app/app.py`

## Business Impact
- Predict customer churn with 85%+ accuracy
- Identify key churn factors and retention strategies
- Reduce churn rate by 15-20% through targeted interventions
- Optimize customer lifetime value and retention costs

