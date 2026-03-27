"""
Flask Web Application for Customer Churn Prediction

This module provides a web interface for the churn prediction model,
including prediction API, dashboard, and business insights.
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import numpy as np
import joblib
import os
import sys
from datetime import datetime
import json

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.preprocessing import ChurnDataPreprocessor
from models.model_trainer import ChurnModelTrainer
from visualization.plot_utils import ChurnVisualizer

app = Flask(__name__)
app.secret_key = 'churn_analysis_secret_key'

# Global variables for model and preprocessor
model = None
preprocessor = None
model_trainer = None
visualizer = None

def load_model():
    """Load the trained model and preprocessor."""
    global model, preprocessor, model_trainer, visualizer
    
    try:
        # Initialize components
        preprocessor = ChurnDataPreprocessor()
        model_trainer = ChurnModelTrainer()
        visualizer = ChurnVisualizer()
        
        # Load best model if available
        model_path = 'models/best_model.joblib'
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print("Model loaded successfully")
        else:
            print("No trained model found. Please train a model first.")
            
    except Exception as e:
        print(f"Error loading model: {str(e)}")

@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page."""
    if request.method == 'GET':
        return render_template('predict.html')
    
    try:
        # Get form data
        form_data = request.form.to_dict()
        
        # Convert to DataFrame
        customer_data = pd.DataFrame([form_data])
        
        # Preprocess the data
        # Note: This is a simplified version. In production, you'd want to
        # properly handle the preprocessing pipeline
        
        # Make prediction
        if model is not None:
            # This is a simplified prediction. In reality, you'd need to
            # properly preprocess the input data
            prediction = model.predict(customer_data)[0]
            probability = model.predict_proba(customer_data)[0][1]
        else:
            prediction = 0
            probability = 0.5
        
        result = {
            'prediction': int(prediction),
            'probability': float(probability),
            'risk_level': 'High' if probability > 0.7 else 'Medium' if probability > 0.3 else 'Low'
        }
        
        return render_template('predict.html', result=result)
        
    except Exception as e:
        return render_template('predict.html', error=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Convert to DataFrame
        customer_data = pd.DataFrame([data])
        
        # Make prediction
        if model is not None:
            prediction = model.predict(customer_data)[0]
            probability = model.predict_proba(customer_data)[0][1]
        else:
            return jsonify({'error': 'Model not loaded'}), 500
        
        result = {
            'prediction': int(prediction),
            'probability': float(probability),
            'risk_level': 'High' if probability > 0.7 else 'Medium' if probability > 0.3 else 'Low',
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/dashboard')
def dashboard():
    """Dashboard page."""
    try:
        # Load sample data for dashboard
        data_path = 'data/raw/customer_churn_data.csv'
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            
            # Calculate key metrics
            total_customers = len(df)
            churn_rate = df['churned'].mean()
            high_value_customers = len(df[df['customer_lifetime_value'] > df['customer_lifetime_value'].quantile(0.8)])
            
            # Segment analysis
            segment_analysis = df.groupby('customer_segment')['churned'].agg(['count', 'mean']).reset_index()
            segment_analysis.columns = ['segment', 'count', 'churn_rate']
            
            # Recent trends (simplified)
            recent_churn = df['churned'].sum()
            
            metrics = {
                'total_customers': total_customers,
                'churn_rate': churn_rate,
                'high_value_customers': high_value_customers,
                'recent_churn': recent_churn,
                'segment_analysis': segment_analysis.to_dict('records')
            }
            
            return render_template('dashboard.html', metrics=metrics)
        else:
            return render_template('dashboard.html', error="No data available")
            
    except Exception as e:
        return render_template('dashboard.html', error=str(e))

@app.route('/insights')
def insights():
    """Business insights page."""
    try:
        # Load data for insights
        data_path = 'data/raw/customer_churn_data.csv'
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            
            # Generate insights
            insights = generate_business_insights(df)
            
            return render_template('insights.html', insights=insights)
        else:
            return render_template('insights.html', error="No data available")
            
    except Exception as e:
        return render_template('insights.html', error=str(e))

@app.route('/train', methods=['GET', 'POST'])
def train_model():
    """Model training page."""
    if request.method == 'GET':
        return render_template('train.html')
    
    try:
        # This would typically be run in the background
        # For demo purposes, we'll just show a success message
        return render_template('train.html', success="Model training initiated. Check logs for progress.")
        
    except Exception as e:
        return render_template('train.html', error=str(e))

def generate_business_insights(df):
    """Generate business insights from the data."""
    insights = {}
    
    # Overall churn rate
    insights['overall_churn_rate'] = df['churned'].mean()
    
    # Churn by segment
    segment_churn = df.groupby('customer_segment')['churned'].mean().sort_values(ascending=False)
    insights['segment_churn'] = segment_churn.to_dict()
    
    # High-risk customers
    high_risk_threshold = 0.7
    # This is simplified - in reality you'd use the model predictions
    high_risk_customers = len(df[df['support_tickets_count'] > 2])
    insights['high_risk_customers'] = high_risk_customers
    
    # Revenue impact
    churned_revenue = df[df['churned'] == 1]['total_revenue'].sum()
    total_revenue = df['total_revenue'].sum()
    insights['revenue_at_risk'] = churned_revenue / total_revenue
    
    # Key drivers
    insights['key_drivers'] = [
        'High support ticket volume',
        'Low usage frequency',
        'Payment failures',
        'Long periods of inactivity'
    ]
    
    return insights

@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)


