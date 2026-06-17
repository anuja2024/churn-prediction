"""
CHURN PREDICTION - STEP 2: PREPROCESSING & FEATURE ENGINEERING
Clean, transform, and prepare data for modeling
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

print("="*70)
print("CHURN PREDICTION - PREPROCESSING & FEATURE ENGINEERING")
print("="*70)

# Load data
print("\n1. LOADING DATA...")
df = pd.read_csv('data/processed/churn_data_raw.csv')
print(f"   ✓ Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")

# Clean column names
df.columns = df.columns.str.strip()

# Check missing values
print("\n2. DATA CLEANING...")
missing = df.isnull().sum().sum()
if missing == 0:
    print(f"   ✓ No missing values!")
else:
    df = df.dropna()
    print(f"   ✓ Removed rows with missing values")

# Remove customerID (not useful)
if 'customerID' in df.columns:
    df = df.drop('customerID', axis=1)
    print(f"   ✓ Removed customerID column")

# Fix TotalCharges column
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges'])
print(f"   ✓ Cleaned TotalCharges column")

# Encode target variable
print("\n3. ENCODING TARGET VARIABLE...")
df['Churn'] = (df['Churn'] == 'Yes').astype(int)
print(f"   ✓ Churn: No=0, Yes=1")

# Encode categorical variables
print("\n4. ENCODING CATEGORICAL VARIABLES...")
categorical_features = df.select_dtypes(include=['object']).columns
label_encoders = {}

for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    print(f"   ✓ {col}")

# Save encoders
joblib.dump(label_encoders, 'models/label_encoders.pkl')
print(f"   ✓ Saved label encoders")

# Separate features and target
print("\n5. SEPARATING FEATURES AND TARGET...")
X = df.drop('Churn', axis=1)
y = df['Churn']

print(f"   Features (X): {X.shape}")
print(f"   Target (y): {y.shape}")
print(f"   Churn rate: {y.mean()*100:.2f}%")

# Train-test split
print("\n6. TRAIN-TEST SPLIT...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

print(f"   Training set: {X_train.shape[0]:,} samples")
print(f"   Test set: {X_test.shape[0]:,} samples")
print(f"   ✓ Stratified split (churn ratio preserved)")

# Scale features
print("\n7. FEATURE SCALING...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

joblib.dump(scaler, 'models/scaler.pkl')
print(f"   ✓ Features scaled (mean=0, std=1)")

# Save processed data
print("\n8. SAVING PROCESSED DATA...")
X_train_scaled.to_csv('data/processed/X_train.csv', index=False)
X_test_scaled.to_csv('data/processed/X_test.csv', index=False)
y_train.to_csv('data/processed/y_train.csv', index=False, header=['Churn'])
y_test.to_csv('data/processed/y_test.csv', index=False, header=['Churn'])

print(f"   ✓ data/processed/X_train.csv")
print(f"   ✓ data/processed/X_test.csv")
print(f"   ✓ data/processed/y_train.csv")
print(f"   ✓ data/processed/y_test.csv")

print("\n" + "="*70)
print("✓ PREPROCESSING COMPLETE!")
print("="*70)
print(f"\nData Summary:")
print(f"   - Original: {df.shape[0]:,} rows")
print(f"   - Features: {X.shape[1]}")
print(f"   - Churn rate: {y.mean()*100:.2f}%")
print(f"\nNext step: Run 03_model_training.py")
print("="*70)