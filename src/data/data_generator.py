"""
Synthetic Customer Churn Data Generator

This module generates realistic customer data for churn prediction analysis.
It simulates a subscription-based business with various customer segments,
usage patterns, and churn behaviors.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Tuple, Dict, List
import os

class CustomerDataGenerator:
    """Generate synthetic customer data with realistic churn patterns."""
    
    def __init__(self, random_seed: int = 42):
        """Initialize the data generator with a random seed for reproducibility."""
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Customer segments and their characteristics
        self.segments = {
            'premium': {'churn_rate': 0.05, 'avg_spend': 150, 'usage_freq': 0.8},
            'standard': {'churn_rate': 0.15, 'avg_spend': 75, 'usage_freq': 0.6},
            'basic': {'churn_rate': 0.25, 'avg_spend': 35, 'usage_freq': 0.4},
            'trial': {'churn_rate': 0.40, 'avg_spend': 0, 'usage_freq': 0.3}
        }
        
        # Subscription plans
        self.plans = ['basic', 'standard', 'premium', 'enterprise']
        
        # Geographic regions with different churn patterns
        self.regions = {
            'North America': {'churn_rate': 0.12, 'avg_support_rating': 4.2},
            'Europe': {'churn_rate': 0.10, 'avg_support_rating': 4.4},
            'Asia': {'churn_rate': 0.18, 'avg_support_rating': 3.8},
            'South America': {'churn_rate': 0.22, 'avg_support_rating': 3.6},
            'Africa': {'churn_rate': 0.25, 'avg_support_rating': 3.5}
        }
    
    def generate_customer_base(self, n_customers: int = 10000) -> pd.DataFrame:
        """Generate base customer information."""
        print(f"Generating {n_customers} customers...")
        
        # Basic customer information
        customer_ids = [f"CUST_{i:06d}" for i in range(1, n_customers + 1)]
        
        # Age distribution (18-80, weighted towards 25-45)
        ages = np.random.normal(35, 12, n_customers)
        ages = np.clip(ages, 18, 80).astype(int)
        
        # Gender distribution
        genders = np.random.choice(['Male', 'Female', 'Other'], n_customers, p=[0.48, 0.48, 0.04])
        
        # Geographic distribution
        regions = np.random.choice(list(self.regions.keys()), n_customers, 
                                 p=[0.35, 0.25, 0.20, 0.15, 0.05])
        
        # Subscription plans (weighted towards basic/standard)
        plans = np.random.choice(self.plans, n_customers, p=[0.4, 0.35, 0.20, 0.05])
        
        # Customer segments (correlated with plan)
        segments = []
        for plan in plans:
            if plan == 'basic':
                segments.append(np.random.choice(['basic', 'trial'], p=[0.7, 0.3]))
            elif plan == 'standard':
                segments.append(np.random.choice(['standard', 'basic'], p=[0.8, 0.2]))
            elif plan == 'premium':
                segments.append(np.random.choice(['premium', 'standard'], p=[0.7, 0.3]))
            else:  # enterprise
                segments.append('premium')
        
        # Account creation dates (last 3 years)
        start_date = datetime.now() - timedelta(days=3*365)
        account_created = [start_date + timedelta(days=np.random.randint(0, 3*365)) 
                          for _ in range(n_customers)]
        
        # Customer lifetime value (CLV) - correlated with segment and plan
        clv = []
        for i, (segment, plan) in enumerate(zip(segments, plans)):
            base_clv = self.segments[segment]['avg_spend']
            # Add some randomness and correlation with age
            age_factor = 1 + (ages[i] - 35) / 100
            plan_multiplier = {'basic': 1, 'standard': 1.5, 'premium': 2.5, 'enterprise': 4}[plan]
            clv.append(max(0, np.random.normal(base_clv * plan_multiplier * age_factor, base_clv * 0.3)))
        
        return pd.DataFrame({
            'customer_id': customer_ids,
            'age': ages,
            'gender': genders,
            'region': regions,
            'subscription_plan': plans,
            'customer_segment': segments,
            'account_created': account_created,
            'customer_lifetime_value': clv
        })
    
    def generate_usage_data(self, customers_df: pd.DataFrame) -> pd.DataFrame:
        """Generate usage patterns and behavioral data."""
        print("Generating usage patterns...")
        
        usage_data = []
        
        for _, customer in customers_df.iterrows():
            # Days since account creation
            days_since_creation = (datetime.now() - customer['account_created']).days
            
            # Usage frequency based on segment
            base_freq = self.segments[customer['customer_segment']]['usage_freq']
            region_factor = 1 - (self.regions[customer['region']]['churn_rate'] - 0.12) * 0.5
            usage_frequency = base_freq * region_factor
            
            # Monthly usage sessions (last 12 months)
            monthly_sessions = []
            for month in range(12):
                # Sessions per month (Poisson distribution)
                avg_sessions = usage_frequency * 30  # 30 days per month
                sessions = np.random.poisson(avg_sessions)
                monthly_sessions.append(sessions)
            
            # Average session duration (minutes)
            avg_session_duration = np.random.normal(25, 10)
            avg_session_duration = max(5, min(120, avg_session_duration))
            
            # Total sessions in last 30 days
            sessions_last_30_days = np.random.poisson(usage_frequency * 30)
            
            # Days since last login
            if sessions_last_30_days > 0:
                days_since_last_login = np.random.randint(1, 30)
            else:
                days_since_last_login = np.random.randint(30, 180)
            
            # Support tickets (correlated with churn risk)
            support_tickets = np.random.poisson(0.5)  # Average 0.5 tickets per customer
            
            # Payment method (affects churn)
            payment_methods = ['credit_card', 'debit_card', 'paypal', 'bank_transfer']
            payment_method = np.random.choice(payment_methods, p=[0.4, 0.3, 0.2, 0.1])
            
            # Device types
            device_types = ['mobile', 'desktop', 'tablet']
            primary_device = np.random.choice(device_types, p=[0.6, 0.35, 0.05])
            
            usage_data.append({
                'customer_id': customer['customer_id'],
                'total_sessions_last_12_months': sum(monthly_sessions),
                'sessions_last_30_days': sessions_last_30_days,
                'avg_session_duration_minutes': avg_session_duration,
                'days_since_last_login': days_since_last_login,
                'support_tickets_count': support_tickets,
                'payment_method': payment_method,
                'primary_device': primary_device,
                'account_age_days': days_since_creation
            })
        
        return pd.DataFrame(usage_data)
    
    def generate_financial_data(self, customers_df: pd.DataFrame) -> pd.DataFrame:
        """Generate financial and billing data."""
        print("Generating financial data...")
        
        financial_data = []
        
        for _, customer in customers_df.iterrows():
            # Monthly revenue (varies by plan)
            plan_prices = {'basic': 9.99, 'standard': 19.99, 'premium': 39.99, 'enterprise': 99.99}
            base_monthly_revenue = plan_prices[customer['subscription_plan']]
            
            # Add some variation
            monthly_revenue = np.random.normal(base_monthly_revenue, base_monthly_revenue * 0.1)
            monthly_revenue = max(0, monthly_revenue)
            
            # Total revenue (based on account age)
            account_age_months = customer['account_created']
            months_since_creation = (datetime.now() - account_age_months).days / 30
            total_revenue = monthly_revenue * months_since_creation
            
            # Payment failures (indicator of churn risk)
            payment_failures = np.random.poisson(0.2)  # Average 0.2 failures per customer
            
            # Discounts received
            discounts_received = np.random.poisson(0.3)
            
            # Refund requests
            refund_requests = np.random.poisson(0.1)
            
            financial_data.append({
                'customer_id': customer['customer_id'],
                'monthly_revenue': monthly_revenue,
                'total_revenue': total_revenue,
                'payment_failures_count': payment_failures,
                'discounts_received_count': discounts_received,
                'refund_requests_count': refund_requests
            })
        
        return pd.DataFrame(financial_data)
    
    def generate_churn_labels(self, customers_df: pd.DataFrame, 
                            usage_df: pd.DataFrame, 
                            financial_df: pd.DataFrame) -> pd.DataFrame:
        """Generate churn labels based on realistic patterns."""
        print("Generating churn labels...")
        
        churn_data = []
        
        for _, customer in customers_df.iterrows():
            # Get related data
            usage = usage_df[usage_df['customer_id'] == customer['customer_id']].iloc[0]
            financial = financial_df[financial_df['customer_id'] == customer['customer_id']].iloc[0]
            
            # Base churn probability from segment and region
            base_churn_prob = self.segments[customer['customer_segment']]['churn_rate']
            region_churn_prob = self.regions[customer['region']]['churn_rate']
            combined_base_prob = (base_churn_prob + region_churn_prob) / 2
            
            # Risk factors that increase churn probability
            risk_factors = 0
            
            # Low usage
            if usage['sessions_last_30_days'] < 5:
                risk_factors += 0.3
            if usage['days_since_last_login'] > 14:
                risk_factors += 0.4
            
            # Support issues
            if usage['support_tickets_count'] > 2:
                risk_factors += 0.2
            
            # Payment issues
            if financial['payment_failures_count'] > 1:
                risk_factors += 0.3
            
            # Account age (new customers more likely to churn)
            if usage['account_age_days'] < 30:
                risk_factors += 0.2
            elif usage['account_age_days'] > 365:
                risk_factors -= 0.1  # Loyal customers less likely to churn
            
            # Calculate final churn probability
            churn_probability = min(0.8, combined_base_prob + risk_factors)
            
            # Generate churn label
            churned = np.random.random() < churn_probability
            
            # Churn date (if churned)
            churn_date = None
            if churned:
                # Churn date within last 6 months
                churn_date = datetime.now() - timedelta(days=np.random.randint(1, 180))
            
            churn_data.append({
                'customer_id': customer['customer_id'],
                'churned': int(churned),
                'churn_probability': churn_probability,
                'churn_date': churn_date,
                'churn_reason': self._get_churn_reason(churned, usage, financial) if churned else None
            })
        
        return pd.DataFrame(churn_data)
    
    def _get_churn_reason(self, churned: bool, usage: pd.Series, financial: pd.Series) -> str:
        """Generate realistic churn reasons."""
        if not churned:
            return None
        
        reasons = []
        
        if usage['sessions_last_30_days'] < 3:
            reasons.append('Low usage')
        if usage['days_since_last_login'] > 30:
            reasons.append('Inactive account')
        if usage['support_tickets_count'] > 3:
            reasons.append('Poor support experience')
        if financial['payment_failures_count'] > 2:
            reasons.append('Payment issues')
        if financial['refund_requests_count'] > 0:
            reasons.append('Dissatisfaction with service')
        
        if not reasons:
            reasons = ['Price sensitivity', 'Found alternative', 'No longer needed']
        
        return ', '.join(reasons[:2])  # Return top 2 reasons
    
    def generate_complete_dataset(self, n_customers: int = 10000) -> pd.DataFrame:
        """Generate the complete customer churn dataset."""
        print("=" * 50)
        print("GENERATING SYNTHETIC CUSTOMER CHURN DATASET")
        print("=" * 50)
        
        # Generate base customer data
        customers_df = self.generate_customer_base(n_customers)
        
        # Generate usage data
        usage_df = self.generate_usage_data(customers_df)
        
        # Generate financial data
        financial_df = self.generate_financial_data(customers_df)
        
        # Generate churn labels
        churn_df = self.generate_churn_labels(customers_df, usage_df, financial_df)
        
        # Combine all data
        print("Combining datasets...")
        final_df = customers_df.merge(usage_df, on='customer_id')
        final_df = final_df.merge(financial_df, on='customer_id')
        final_df = final_df.merge(churn_df, on='customer_id')
        
        # Add some additional derived features
        final_df['revenue_per_session'] = final_df['total_revenue'] / (final_df['total_sessions_last_12_months'] + 1)
        final_df['usage_intensity'] = final_df['sessions_last_30_days'] / 30
        final_df['risk_score'] = (
            (final_df['support_tickets_count'] > 2).astype(int) +
            (final_df['payment_failures_count'] > 1).astype(int) +
            (final_df['days_since_last_login'] > 14).astype(int) +
            (final_df['sessions_last_30_days'] < 5).astype(int)
        )
        
        print(f"Dataset generated successfully!")
        print(f"Total customers: {len(final_df)}")
        print(f"Churn rate: {final_df['churned'].mean():.2%}")
        print(f"Features: {len(final_df.columns)}")
        
        return final_df

def main():
    """Generate and save the synthetic dataset."""
    generator = CustomerDataGenerator(random_seed=42)
    
    # Generate dataset
    df = generator.generate_complete_dataset(n_customers=10000)
    
    # Save to CSV
    os.makedirs('data/raw', exist_ok=True)
    output_path = 'data/raw/customer_churn_data.csv'
    df.to_csv(output_path, index=False)
    
    print(f"\nDataset saved to: {output_path}")
    print("\nDataset Summary:")
    print(df.describe())
    
    return df

if __name__ == "__main__":
    df = main()

