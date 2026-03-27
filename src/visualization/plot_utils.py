"""
Visualization Utilities for Customer Churn Analysis

This module provides comprehensive visualization functions for
exploratory data analysis and model evaluation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ChurnVisualizer:
    """Comprehensive visualization class for churn analysis."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """Initialize the visualizer."""
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
    def plot_churn_distribution(self, df: pd.DataFrame, save_path: str = None):
        """Plot churn distribution and basic statistics."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Churn count
        churn_counts = df['churned'].value_counts()
        axes[0].pie(churn_counts.values, labels=['Not Churned', 'Churned'], 
                   autopct='%1.1f%%', colors=['#2ca02c', '#d62728'])
        axes[0].set_title('Churn Distribution')
        
        # Churn rate by segment
        churn_by_segment = df.groupby('customer_segment')['churned'].mean().sort_values(ascending=False)
        churn_by_segment.plot(kind='bar', ax=axes[1], color=self.colors[:len(churn_by_segment)])
        axes[1].set_title('Churn Rate by Customer Segment')
        axes[1].set_ylabel('Churn Rate')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_demographic_analysis(self, df: pd.DataFrame, save_path: str = None):
        """Plot demographic analysis of churn."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Age distribution by churn
        df.boxplot(column='age', by='churned', ax=axes[0, 0])
        axes[0, 0].set_title('Age Distribution by Churn Status')
        axes[0, 0].set_xlabel('Churned')
        axes[0, 0].set_ylabel('Age')
        
        # Gender and churn
        gender_churn = pd.crosstab(df['gender'], df['churned'], normalize='index')
        gender_churn.plot(kind='bar', ax=axes[0, 1], color=['#2ca02c', '#d62728'])
        axes[0, 1].set_title('Churn Rate by Gender')
        axes[0, 1].set_ylabel('Churn Rate')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Region and churn
        region_churn = pd.crosstab(df['region'], df['churned'], normalize='index')
        region_churn.plot(kind='bar', ax=axes[1, 0], color=['#2ca02c', '#d62728'])
        axes[1, 0].set_title('Churn Rate by Region')
        axes[1, 0].set_ylabel('Churn Rate')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Subscription plan and churn
        plan_churn = pd.crosstab(df['subscription_plan'], df['churned'], normalize='index')
        plan_churn.plot(kind='bar', ax=axes[1, 1], color=['#2ca02c', '#d62728'])
        axes[1, 1].set_title('Churn Rate by Subscription Plan')
        axes[1, 1].set_ylabel('Churn Rate')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_usage_patterns(self, df: pd.DataFrame, save_path: str = None):
        """Plot usage patterns and their relationship to churn."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Sessions vs churn
        df.boxplot(column='sessions_last_30_days', by='churned', ax=axes[0, 0])
        axes[0, 0].set_title('Sessions Last 30 Days by Churn Status')
        axes[0, 0].set_xlabel('Churned')
        axes[0, 0].set_ylabel('Sessions')
        
        # Session duration vs churn
        df.boxplot(column='avg_session_duration_minutes', by='churned', ax=axes[0, 1])
        axes[0, 1].set_title('Average Session Duration by Churn Status')
        axes[0, 1].set_xlabel('Churned')
        axes[0, 1].set_ylabel('Duration (minutes)')
        
        # Days since last login
        df.boxplot(column='days_since_last_login', by='churned', ax=axes[1, 0])
        axes[1, 0].set_title('Days Since Last Login by Churn Status')
        axes[1, 0].set_xlabel('Churned')
        axes[1, 0].set_ylabel('Days')
        
        # Support tickets
        df.boxplot(column='support_tickets_count', by='churned', ax=axes[1, 1])
        axes[1, 1].set_title('Support Tickets by Churn Status')
        axes[1, 1].set_xlabel('Churned')
        axes[1, 1].set_ylabel('Ticket Count')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_financial_analysis(self, df: pd.DataFrame, save_path: str = None):
        """Plot financial metrics and their relationship to churn."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Monthly revenue vs churn
        df.boxplot(column='monthly_revenue', by='churned', ax=axes[0, 0])
        axes[0, 0].set_title('Monthly Revenue by Churn Status')
        axes[0, 0].set_xlabel('Churned')
        axes[0, 0].set_ylabel('Monthly Revenue ($)')
        
        # Total revenue vs churn
        df.boxplot(column='total_revenue', by='churned', ax=axes[0, 1])
        axes[0, 1].set_title('Total Revenue by Churn Status')
        axes[0, 1].set_xlabel('Churned')
        axes[0, 1].set_ylabel('Total Revenue ($)')
        
        # Payment failures
        df.boxplot(column='payment_failures_count', by='churned', ax=axes[1, 0])
        axes[1, 0].set_title('Payment Failures by Churn Status')
        axes[1, 0].set_xlabel('Churned')
        axes[1, 0].set_ylabel('Failure Count')
        
        # CLV vs churn
        df.boxplot(column='customer_lifetime_value', by='churned', ax=axes[1, 1])
        axes[1, 1].set_title('Customer Lifetime Value by Churn Status')
        axes[1, 1].set_xlabel('Churned')
        axes[1, 1].set_ylabel('CLV ($)')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_heatmap(self, df: pd.DataFrame, save_path: str = None):
        """Plot correlation heatmap of numerical features."""
        # Select numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove customer_id if present
        if 'customer_id' in numerical_cols:
            numerical_cols.remove('customer_id')
        
        # Calculate correlation matrix
        corr_matrix = df[numerical_cols].corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8})
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, feature_importance: pd.DataFrame, 
                              top_n: int = 20, save_path: str = None):
        """Plot feature importance from tree-based models."""
        # Get top N features
        top_features = feature_importance.head(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_interactive_dashboard(self, df: pd.DataFrame, save_path: str = None):
        """Create an interactive dashboard using Plotly."""
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Churn Distribution', 'Churn by Segment',
                'Age Distribution', 'Usage Patterns',
                'Revenue Analysis', 'Geographic Distribution'
            ],
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "histogram"}, {"type": "scatter"}],
                   [{"type": "box"}, {"type": "bar"}]]
        )
        
        # Churn distribution pie chart
        churn_counts = df['churned'].value_counts()
        fig.add_trace(
            go.Pie(labels=['Not Churned', 'Churned'], values=churn_counts.values),
            row=1, col=1
        )
        
        # Churn by segment
        segment_churn = df.groupby('customer_segment')['churned'].mean()
        fig.add_trace(
            go.Bar(x=segment_churn.index, y=segment_churn.values),
            row=1, col=2
        )
        
        # Age distribution
        fig.add_trace(
            go.Histogram(x=df['age'], nbinsx=30),
            row=2, col=1
        )
        
        # Usage patterns scatter
        fig.add_trace(
            go.Scatter(
                x=df['sessions_last_30_days'],
                y=df['avg_session_duration_minutes'],
                mode='markers',
                marker=dict(color=df['churned'], colorscale='RdYlGn')
            ),
            row=2, col=2
        )
        
        # Revenue analysis
        fig.add_trace(
            go.Box(y=df['monthly_revenue'], x=df['churned']),
            row=3, col=1
        )
        
        # Geographic distribution
        region_churn = df.groupby('region')['churned'].mean()
        fig.add_trace(
            go.Bar(x=region_churn.index, y=region_churn.values),
            row=3, col=2
        )
        
        fig.update_layout(height=1200, showlegend=False)
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
    
    def plot_time_series_analysis(self, df: pd.DataFrame, save_path: str = None):
        """Plot time series analysis of churn patterns."""
        # Convert account_created to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df['account_created']):
            df['account_created'] = pd.to_datetime(df['account_created'])
        
        # Create monthly churn analysis
        df['account_month'] = df['account_created'].dt.to_period('M')
        monthly_churn = df.groupby('account_month')['churned'].agg(['count', 'sum', 'mean']).reset_index()
        monthly_churn['churn_rate'] = monthly_churn['mean']
        monthly_churn['total_customers'] = monthly_churn['count']
        monthly_churn['churned_customers'] = monthly_churn['sum']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Monthly churn rate
        monthly_churn.plot(x='account_month', y='churn_rate', ax=axes[0, 0], marker='o')
        axes[0, 0].set_title('Monthly Churn Rate Over Time')
        axes[0, 0].set_ylabel('Churn Rate')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Total customers by month
        monthly_churn.plot(x='account_month', y='total_customers', ax=axes[0, 1], marker='o')
        axes[0, 1].set_title('Total Customers by Month')
        axes[0, 1].set_ylabel('Customer Count')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Churned customers by month
        monthly_churn.plot(x='account_month', y='churned_customers', ax=axes[1, 0], marker='o')
        axes[1, 0].set_title('Churned Customers by Month')
        axes[1, 0].set_ylabel('Churned Count')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Account age vs churn rate
        df['account_age_days'] = (pd.Timestamp.now() - df['account_created']).dt.days
        df['account_age_months'] = df['account_age_days'] / 30
        
        # Bin by account age
        df['age_bin'] = pd.cut(df['account_age_months'], bins=12, labels=False)
        age_churn = df.groupby('age_bin')['churned'].mean()
        
        age_churn.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Churn Rate by Account Age')
        axes[1, 1].set_ylabel('Churn Rate')
        axes[1, 1].set_xlabel('Account Age (months)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_performance_comparison(self, model_results: Dict, save_path: str = None):
        """Plot model performance comparison."""
        # Extract metrics
        models = list(model_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
        
        # Create comparison data
        comparison_data = []
        for model in models:
            row = {'Model': model}
            for metric in metrics:
                if metric in model_results[model]['metrics']:
                    row[metric] = model_results[model]['metrics'][metric]
            comparison_data.append(row)
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if i < len(axes):
                df_comparison.plot(x='Model', y=metric, kind='bar', ax=axes[i], 
                                 color=self.colors[i % len(self.colors)])
                axes[i].set_title(f'{metric.upper()} Comparison')
                axes[i].set_ylabel(metric.title())
                axes[i].tick_params(axis='x', rotation=45)
        
        # Remove empty subplot
        if len(metrics) < len(axes):
            axes[-1].remove()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_comprehensive_report(self, df: pd.DataFrame, model_results: Dict = None, 
                                 save_dir: str = 'reports'):
        """Create a comprehensive visualization report."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print("Creating comprehensive visualization report...")
        
        # Basic churn analysis
        self.plot_churn_distribution(df, f'{save_dir}/churn_distribution.png')
        self.plot_demographic_analysis(df, f'{save_dir}/demographic_analysis.png')
        self.plot_usage_patterns(df, f'{save_dir}/usage_patterns.png')
        self.plot_financial_analysis(df, f'{save_dir}/financial_analysis.png')
        self.plot_correlation_heatmap(df, f'{save_dir}/correlation_heatmap.png')
        self.plot_time_series_analysis(df, f'{save_dir}/time_series_analysis.png')
        
        # Interactive dashboard
        self.create_interactive_dashboard(df, f'{save_dir}/interactive_dashboard.html')
        
        # Model performance (if available)
        if model_results:
            self.plot_model_performance_comparison(model_results, f'{save_dir}/model_performance.png')
        
        print(f"Visualization report saved to {save_dir}/")

def main():
    """Example usage of the visualizer."""
    from src.data.data_generator import CustomerDataGenerator
    
    # Generate sample data
    generator = CustomerDataGenerator()
    df = generator.generate_complete_dataset(n_customers=5000)
    
    # Create visualizations
    visualizer = ChurnVisualizer()
    visualizer.create_comprehensive_report(df)
    
    return visualizer

if __name__ == "__main__":
    visualizer = main()


