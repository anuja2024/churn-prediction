"""
CHURN PREDICTION - STEP 1: DATA LOADING & EXPLORATION
Load and explore the telecom customer churn dataset
"""

import pandas as pd
import numpy as np
import os

# Create directories
os.makedirs('data/processed', exist_ok=True)

print("="*70)
print("CHURN PREDICTION - DATA LOADING & EXPLORATION")
print("="*70)

# Load the dataset
print("\n1. LOADING DATA...")

df = pd.read_csv('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')

print(f"   ✓ Loaded! Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

# Show basic info
print("\n2. BASIC INFORMATION")
print("-" * 70)

print(f"\nDataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

print(f"\nColumn names:")
for i, col in enumerate(df.columns, 1):
    print(f"   {i}. {col}")

print(f"\nMissing values: {df.isnull().sum().sum()}")

# Churn analysis
print("\n3. CHURN ANALYSIS")
print("-" * 70)

churn_counts = df['Churn'].value_counts()
churn_pcts = df['Churn'].value_counts(normalize=True) * 100

print(f"\nChurn Distribution:")
print(f"   No:  {churn_counts['No']:,} customers ({churn_pcts['No']:.1f}%)")
print(f"   Yes: {churn_counts['Yes']:,} customers ({churn_pcts['Yes']:.1f}%)")

# Show first few rows
print("\n4. FIRST FEW ROWS")
print("-" * 70)
print("\n" + df.head().to_string())

# Basic statistics
print("\n5. STATISTICS")
print("-" * 70)
print("\n" + df.describe().to_string())

# Save processed data
df.to_csv('data/processed/churn_data_raw.csv', index=False)

print("\n" + "="*70)
print("✓ DATA LOADING COMPLETE!")
print("="*70)
print(f"\n✓ Data saved to: data/processed/churn_data_raw.csv")
print("\nNext step: Run 02_preprocessing.py")
print("="*70)