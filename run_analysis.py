#!/usr/bin/env python3
"""
Main script to run the complete customer churn analysis pipeline.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append('src')

def main():
    """Run the complete churn analysis pipeline."""
    print("=" * 60)
    print("CUSTOMER CHURN PREDICTION ANALYSIS")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    
    try:
        # Step 1: Generate synthetic data
        print("\n1. Generating synthetic customer data...")
        from data.data_generator import CustomerDataGenerator
        
        generator = CustomerDataGenerator(random_seed=42)
        df = generator.generate_complete_dataset(n_customers=5000)  # Smaller dataset for demo
        
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
        
        # Step 4: Preprocess data
        print("\n3. Preprocessing data...")
        from data.preprocessing import ChurnDataPreprocessor
        
        preprocessor = ChurnDataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(df)
        
        print(f"✓ Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"✓ Test set: {X_test.shape[0]} samples")
        print(f"✓ Training churn rate: {y_train.mean():.2%}")
        print(f"✓ Test churn rate: {y_test.mean():.2%}")
        
        # Step 5: Train models
        print("\n4. Training machine learning models...")
        from models.model_trainer import ChurnModelTrainer
        
        trainer = ChurnModelTrainer()
        results = trainer.train_all_models(X_train, y_train, X_test, y_test)
        
        # Get model comparison
        comparison = trainer.get_model_comparison()
        print("\n   Model Performance:")
        print(comparison.to_string(index=False))
        
        # Step 6: Evaluate models
        print("\n5. Evaluating models...")
        from models.model_evaluator import ChurnModelEvaluator
        
        evaluator = ChurnModelEvaluator()
        
        # Prepare results for evaluation
        models_results = {}
        for model_name, metrics in trainer.models.items():
            models_results[model_name] = {
                'y_true': y_test,
                'y_pred': metrics['y_pred'],
                'y_pred_proba': metrics['y_pred_proba'],
                'metrics': metrics
            }
        
        # Generate evaluation report
        report = evaluator.generate_evaluation_report(models_results)
        print("\n" + "="*50)
        print("EVALUATION REPORT")
        print("="*50)
        print(report)
        
        # Step 7: Create visualizations
        print("\n6. Creating visualizations...")
        from visualization.plot_utils import ChurnVisualizer
        
        visualizer = ChurnVisualizer()
        os.makedirs('reports', exist_ok=True)
        visualizer.create_comprehensive_report(df, models_results, save_dir='reports')
        print("✓ Visualizations saved to reports/")
        
        # Step 8: Save models
        print("\n7. Saving models...")
        trainer.save_models()
        print("✓ Models saved to models/")
        
        # Step 9: Generate business insights
        print("\n8. Generating business insights...")
        
        # High-risk customers
        high_risk_customers = len(df[df['support_tickets_count'] > 2])
        
        # Key insights
        insights = {
            'overall_churn_rate': churn_rate,
            'high_risk_customers': high_risk_customers,
            'revenue_at_risk': revenue_at_risk,
            'best_model': trainer.best_model_name,
            'best_score': trainer.best_score
        }
        
        print(f"\n   Key Insights:")
        print(f"   - Overall churn rate: {insights['overall_churn_rate']:.2%}")
        print(f"   - High-risk customers: {insights['high_risk_customers']:,}")
        print(f"   - Revenue at risk: {insights['revenue_at_risk']:.2%}")
        print(f"   - Best model: {insights['best_model']} (AUC: {insights['best_score']:.3f})")
        
        # Step 10: Recommendations
        print(f"\n9. Strategic Recommendations:")
        print(f"   1. Focus retention efforts on high-risk segments")
        print(f"   2. Implement usage monitoring and alerts")
        print(f"   3. Improve customer support response times")
        print(f"   4. Develop proactive engagement campaigns")
        print(f"   5. Create customer health scoring system")
        
        print(f"\n" + "="*60)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Completed at: {datetime.now()}")
        print(f"\nNext steps:")
        print(f"- Review reports/ for visualizations")
        print(f"- Check models/ for trained models")
        print(f"- Run 'python app/app.py' to start web interface")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Analysis completed successfully!")
    else:
        print("\n💥 Analysis failed. Check the error messages above.")
        sys.exit(1)


