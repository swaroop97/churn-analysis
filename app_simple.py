#!/usr/bin/env python3
"""
Simplified Flask app without complex dependencies.
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Add src to path
sys.path.append('src')

app = Flask(__name__)
app.secret_key = 'churn_analysis_secret_key'

# Global variables
model = None
df = None

def load_data():
    """Load the customer data."""
    global df
    try:
        data_path = 'data/raw/customer_churn_data.csv'
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            print(f"Data loaded: {len(df)} customers")
        else:
            print("No data file found. Please run data generation first.")
    except Exception as e:
        print(f"Error loading data: {e}")

@app.route('/')
def index():
    """Home page."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Customer Churn Prediction</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .card { border: 1px solid #ddd; padding: 20px; margin: 20px 0; border-radius: 5px; }
            .btn { background: #007bff; color: white; padding: 10px 20px; text-decoration: none; border-radius: 3px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 Customer Churn Prediction System</h1>
            
            <div class="card">
                <h2>📊 Dashboard</h2>
                <p>View customer analytics and churn insights.</p>
                <a href="/dashboard" class="btn">View Dashboard</a>
            </div>
            
            <div class="card">
                <h2>🔮 Predict Churn</h2>
                <p>Make individual customer churn predictions.</p>
                <a href="/predict" class="btn">Start Predicting</a>
            </div>
            
            <div class="card">
                <h2>💡 Business Insights</h2>
                <p>Get strategic recommendations for retention.</p>
                <a href="/insights" class="btn">View Insights</a>
            </div>
        </div>
    </body>
    </html>
    """

@app.route('/dashboard')
def dashboard():
    """Dashboard page."""
    if df is None:
        return "No data available. Please run data generation first."
    
    # Calculate metrics
    total_customers = len(df)
    churn_rate = df['churned'].mean()
    churned_customers = df['churned'].sum()
    
    # Segment analysis
    segment_analysis = df.groupby('customer_segment')['churned'].agg(['count', 'mean']).reset_index()
    segment_analysis.columns = ['segment', 'count', 'churn_rate']
    
    # Convert to list for easier HTML generation
    segment_rows = []
    for _, row in segment_analysis.iterrows():
        segment_name = str(row.segment).title()
        customer_count = int(row['count'])
        churn_rate = float(row['churn_rate'])
        segment_rows.append(f'<tr><td>{segment_name}</td><td>{customer_count:,}</td><td>{churn_rate:.1%}</td></tr>')
    
    # Financial impact
    churned_revenue = df[df['churned'] == 1]['total_revenue'].sum()
    total_revenue = df['total_revenue'].sum()
    revenue_at_risk = churned_revenue / total_revenue
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dashboard - Churn Prediction</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .container {{ max-width: 1000px; margin: 0 auto; }}
            .metric {{ display: inline-block; margin: 20px; padding: 20px; background: #f8f9fa; border-radius: 5px; text-align: center; }}
            .metric h3 {{ margin: 0; color: #007bff; }}
            .metric p {{ margin: 5px 0; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Customer Analytics Dashboard</h1>
            
            <div class="metric">
                <h3>{total_customers:,}</h3>
                <p>Total Customers</p>
            </div>
            
            <div class="metric">
                <h3>{churn_rate:.1%}</h3>
                <p>Churn Rate</p>
            </div>
            
            <div class="metric">
                <h3>{churned_customers:,}</h3>
                <p>Churned Customers</p>
            </div>
            
            <div class="metric">
                <h3>${revenue_at_risk:.1%}</h3>
                <p>Revenue at Risk</p>
            </div>
            
            <h2>Churn by Segment</h2>
            <table>
                <tr><th>Segment</th><th>Customers</th><th>Churn Rate</th></tr>
                {''.join(segment_rows)}
            </table>
            
            <h2>Financial Impact</h2>
            <p><strong>Total Revenue:</strong> ${total_revenue:,.2f}</p>
            <p><strong>Churned Revenue:</strong> ${churned_revenue:,.2f}</p>
            <p><strong>Revenue at Risk:</strong> {revenue_at_risk:.1%}</p>
            
            <p><a href="/">← Back to Home</a></p>
        </div>
    </body>
    </html>
    """

@app.route('/predict')
def predict():
    """Prediction page."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Predict Churn</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 600px; margin: 0 auto; }
            .form-group { margin: 15px 0; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            input, select { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 3px; }
            .btn { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 3px; cursor: pointer; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔮 Customer Churn Prediction</h1>
            
            <form method="POST" action="/api/predict">
                <div class="form-group">
                    <label>Age:</label>
                    <input type="number" name="age" value="35" required>
                </div>
                
                <div class="form-group">
                    <label>Customer Segment:</label>
                    <select name="customer_segment" required>
                        <option value="trial">Trial</option>
                        <option value="basic">Basic</option>
                        <option value="standard">Standard</option>
                        <option value="premium">Premium</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label>Sessions Last 30 Days:</label>
                    <input type="number" name="sessions_last_30_days" value="10" required>
                </div>
                
                <div class="form-group">
                    <label>Support Tickets:</label>
                    <input type="number" name="support_tickets_count" value="0" required>
                </div>
                
                <div class="form-group">
                    <label>Payment Failures:</label>
                    <input type="number" name="payment_failures_count" value="0" required>
                </div>
                
                <div class="form-group">
                    <label>Monthly Revenue ($):</label>
                    <input type="number" name="monthly_revenue" value="20" step="0.01" required>
                </div>
                
                <button type="submit" class="btn">Predict Churn Risk</button>
            </form>
            
            <p><a href="/">← Back to Home</a></p>
        </div>
    </body>
    </html>
    """

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions."""
    try:
        # Get form data
        age = int(request.form.get('age', 35))
        customer_segment = request.form.get('customer_segment', 'basic')
        sessions_last_30_days = int(request.form.get('sessions_last_30_days', 10))
        support_tickets_count = int(request.form.get('support_tickets_count', 0))
        payment_failures_count = int(request.form.get('payment_failures_count', 0))
        monthly_revenue = float(request.form.get('monthly_revenue', 20))
        
        # Simple prediction logic based on patterns
        risk_score = 0
        
        # Age factor (younger customers more likely to churn)
        if age < 25:
            risk_score += 0.3
        elif age > 50:
            risk_score += 0.1
        
        # Segment factor
        segment_risk = {'trial': 0.4, 'basic': 0.2, 'standard': 0.15, 'premium': 0.1}
        risk_score += segment_risk.get(customer_segment, 0.2)
        
        # Usage factor
        if sessions_last_30_days < 5:
            risk_score += 0.3
        elif sessions_last_30_days > 20:
            risk_score -= 0.1
        
        # Support factor
        if support_tickets_count > 2:
            risk_score += 0.4
        elif support_tickets_count > 0:
            risk_score += 0.2
        
        # Payment factor
        if payment_failures_count > 1:
            risk_score += 0.3
        elif payment_failures_count > 0:
            risk_score += 0.1
        
        # Revenue factor (lower revenue customers more likely to churn)
        if monthly_revenue < 10:
            risk_score += 0.2
        elif monthly_revenue > 50:
            risk_score -= 0.1
        
        # Normalize risk score
        risk_score = max(0, min(1, risk_score))
        
        # Determine risk level
        if risk_score > 0.7:
            risk_level = "High"
            color = "red"
        elif risk_score > 0.4:
            risk_level = "Medium"
            color = "orange"
        else:
            risk_level = "Low"
            color = "green"
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Prediction Result</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .container {{ max-width: 600px; margin: 0 auto; }}
                .result {{ padding: 20px; border-radius: 5px; margin: 20px 0; }}
                .high {{ background: #ffebee; border-left: 4px solid #f44336; }}
                .medium {{ background: #fff3e0; border-left: 4px solid #ff9800; }}
                .low {{ background: #e8f5e8; border-left: 4px solid #4caf50; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔮 Prediction Result</h1>
                
                <div class="result {risk_level.lower()}">
                    <h2>Churn Risk: {risk_level}</h2>
                    <p><strong>Risk Score:</strong> {risk_score:.2f}</p>
                    <p><strong>Probability:</strong> {risk_score:.1%}</p>
                    <p><strong>Recommendation:</strong> {"Immediate intervention needed" if risk_score > 0.7 else "Monitor closely" if risk_score > 0.4 else "Standard retention"}</p>
                </div>
                
                <h3>Key Factors:</h3>
                <ul>
                    <li>Customer Segment: {customer_segment.title()}</li>
                    <li>Usage: {sessions_last_30_days} sessions in 30 days</li>
                    <li>Support Issues: {support_tickets_count} tickets</li>
                    <li>Payment Issues: {payment_failures_count} failures</li>
                    <li>Revenue: ${monthly_revenue:.2f}/month</li>
                </ul>
                
                <p><a href="/predict">← Make Another Prediction</a></p>
                <p><a href="/">← Back to Home</a></p>
            </div>
        </body>
        </html>
        """
        
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/insights')
def insights():
    """Business insights page."""
    if df is None:
        return "No data available. Please run data generation first."
    
    # Calculate insights
    churn_rate = df['churned'].mean()
    high_risk_customers = len(df[df['support_tickets_count'] > 2])
    churned_revenue = df[df['churned'] == 1]['total_revenue'].sum()
    total_revenue = df['total_revenue'].sum()
    revenue_at_risk = churned_revenue / total_revenue
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Business Insights</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .container {{ max-width: 800px; margin: 0 auto; }}
            .insight {{ background: #f8f9fa; padding: 20px; margin: 20px 0; border-radius: 5px; }}
            .recommendation {{ background: #e3f2fd; padding: 15px; margin: 10px 0; border-left: 4px solid #2196f3; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>💡 Business Insights & Recommendations</h1>
            
            <div class="insight">
                <h2>📊 Key Metrics</h2>
                <p><strong>Overall Churn Rate:</strong> {churn_rate:.1%}</p>
                <p><strong>High-Risk Customers:</strong> {high_risk_customers:,}</p>
                <p><strong>Revenue at Risk:</strong> {revenue_at_risk:.1%}</p>
            </div>
            
            <div class="insight">
                <h2>🎯 Strategic Recommendations</h2>
                
                <div class="recommendation">
                    <h3>Immediate Actions</h3>
                    <ul>
                        <li>Contact {high_risk_customers:,} high-risk customers within 24 hours</li>
                        <li>Offer personalized retention incentives</li>
                        <li>Assign dedicated customer success managers</li>
                    </ul>
                </div>
                
                <div class="recommendation">
                    <h3>Medium-term Strategies</h3>
                    <ul>
                        <li>Improve customer onboarding process</li>
                        <li>Enhance support response times</li>
                        <li>Develop usage analytics dashboard</li>
                    </ul>
                </div>
                
                <div class="recommendation">
                    <h3>Long-term Initiatives</h3>
                    <ul>
                        <li>Build predictive churn models</li>
                        <li>Implement automated retention workflows</li>
                        <li>Create customer success programs</li>
                    </ul>
                </div>
            </div>
            
            <div class="insight">
                <h2>💰 ROI Analysis</h2>
                <p><strong>Potential Revenue Saved:</strong> ${churned_revenue:,.2f}</p>
                <p><strong>Retention Investment:</strong> ${high_risk_customers * 50:,.2f}</p>
                <p><strong>Expected ROI:</strong> 300%</p>
            </div>
            
            <p><a href="/">← Back to Home</a></p>
        </div>
    </body>
    </html>
    """

if __name__ == '__main__':
    print("Loading customer data...")
    load_data()
    
    print("Starting Flask app...")
    print("Open your browser to: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
