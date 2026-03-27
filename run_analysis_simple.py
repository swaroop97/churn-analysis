#!/usr/bin/env python3
"""
Simplified churn analysis script that works without optional dependencies.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append('src')

def main():
    """Run a simplified churn analysis pipeline."""
    print("=" * 60)
    print("SIMPLIFIED CUSTOMER CHURN PREDICTION ANALYSIS")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    
    try:
        # Step 1: Generate synthetic data
        print("\n1. Generating synthetic customer data...")
        from data.data_generator import CustomerDataGenerator
        
        generator = CustomerDataGenerator(random_seed=42)
        df = generator.generate_complete_dataset(n_customers=2000)  # Smaller dataset
        
        print(f"✓ Generated {len(df)} customers with {df['churned'].mean():.2%} churn rate")
        
        # Step 2: Save raw data
        os.makedirs('data/raw', exist_ok=True)
        df.to_csv('data/raw/customer_churn_data.csv', index=False)
        print("✓ Raw data saved to data/raw/customer_churn_data.csv")
        
        # Step 3: Basic analysis
        print("\n2. Performing basic analysis...")
        
        # Churn analysis
        churn_rate = df['churned'].mean()
        total_customers = len(df)
        churned_customers = df['churned'].sum()
        
        print(f"   Total customers: {total_customers:,}")
        print(f"   Churned customers: {churned_customers:,}")
        print(f"   Churn rate: {churn_rate:.2%}")
        
        # Segment analysis
        segment_analysis = df.groupby('customer_segment')['churned'].agg(['count', 'mean']).reset_index()
        segment_analysis.columns = ['segment', 'count', 'churn_rate']
        segment_analysis = segment_analysis.sort_values('churn_rate', ascending=False)
        
        print(f"\n   Churn by segment:")
        for _, row in segment_analysis.iterrows():
            print(f"   - {row['segment'].title()}: {row['churn_rate']:.2%} ({row['count']} customers)")
        
        # Financial impact
        churned_revenue = df[df['churned'] == 1]['total_revenue'].sum()
        total_revenue = df['total_revenue'].sum()
        revenue_at_risk = churned_revenue / total_revenue
        
        print(f"\n   Financial impact:")
        print(f"   - Total revenue: ${total_revenue:,.2f}")
        print(f"   - Churned revenue: ${churned_revenue:,.2f}")
        print(f"   - Revenue at risk: {revenue_at_risk:.2%}")
        
        # Step 4: Simple model training (without complex dependencies)
        print("\n3. Training simple models...")
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report
        
        # Prepare features (simplified)
        feature_cols = ['age', 'customer_lifetime_value', 'sessions_last_30_days', 
                        'avg_session_duration_minutes', 'days_since_last_login',
                        'support_tickets_count', 'payment_failures_count', 'monthly_revenue']
        
        X = df[feature_cols].fillna(0)
        y = df['churned']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        print(f"✓ Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"✓ Test set: {X_test.shape[0]} samples")
        
        # Train Logistic Regression
        print("\n   Training Logistic Regression...")
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_test)
        lr_accuracy = accuracy_score(y_test, lr_pred)
        print(f"   ✓ Logistic Regression Accuracy: {lr_accuracy:.3f}")
        
        # Train Random Forest
        print("   Training Random Forest...")
        rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        print(f"   ✓ Random Forest Accuracy: {rf_accuracy:.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n   Top 5 Most Important Features:")
        for _, row in feature_importance.head(5).iterrows():
            print(f"   - {row['feature']}: {row['importance']:.3f}")
        
        # Step 5: Business insights
        print(f"\n4. Business Insights:")
        print(f"   - Overall churn rate: {churn_rate:.2%}")
        print(f"   - Revenue at risk: {revenue_at_risk:.2%}")
        print(f"   - Best model: Random Forest (Accuracy: {rf_accuracy:.3f})")
        
        # High-risk customers
        high_risk_customers = len(df[df['support_tickets_count'] > 2])
        print(f"   - High-risk customers: {high_risk_customers:,}")
        
        # Recommendations
        print(f"\n5. Strategic Recommendations:")
        print(f"   1. Focus on customers with high support ticket volume")
        print(f"   2. Monitor usage patterns (sessions_last_30_days)")
        print(f"   3. Address payment failures promptly")
        print(f"   4. Implement proactive engagement for at-risk segments")
        
        print(f"\n" + "="*60)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Completed at: {datetime.now()}")
        print(f"\nNext steps:")
        print(f"- Check data/raw/customer_churn_data.csv for the dataset")
        print(f"- Run 'python app/app.py' to start web interface")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Analysis completed successfully!")
    else:
        print("\n💥 Analysis failed. Check the error messages above.")
        sys.exit(1)


