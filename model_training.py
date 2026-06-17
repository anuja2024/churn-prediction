"""
CHURN PREDICTION - STEP 3: MODEL TRAINING
Train 3 different classification models
- Logistic Regression
- Random Forest
- XGBoost
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
import time

print("="*70)
print("CHURN PREDICTION - MODEL TRAINING")
print("="*70)

# Load data
print("\n1. LOADING PREPROCESSED DATA...")
X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')
y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()

print(f"   Training set: {X_train.shape}")
print(f"   Test set: {X_test.shape}")

# MODEL 1: LOGISTIC REGRESSION
print("\n2. TRAINING LOGISTIC REGRESSION...")
start_time = time.time()

lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
lr.fit(X_train, y_train)
train_time = time.time() - start_time

train_score_lr = lr.score(X_train, y_train)
test_score_lr = lr.score(X_test, y_test)

print(f"   Training time: {train_time:.2f} seconds")
print(f"   Train accuracy: {train_score_lr:.4f}")
print(f"   Test accuracy: {test_score_lr:.4f}")

joblib.dump(lr, 'models/logistic_regression.pkl')
print(f"   ✓ Saved model")

# MODEL 2: RANDOM FOREST
print("\n3. TRAINING RANDOM FOREST...")
start_time = time.time()

rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
train_time = time.time() - start_time

train_score_rf = rf.score(X_train, y_train)
test_score_rf = rf.score(X_test, y_test)

print(f"   Training time: {train_time:.2f} seconds")
print(f"   Train accuracy: {train_score_rf:.4f}")
print(f"   Test accuracy: {test_score_rf:.4f}")

joblib.dump(rf, 'models/random_forest.pkl')
print(f"   ✓ Saved model")

# MODEL 3: XGBOOST
print("\n4. TRAINING XGBOOST...")
start_time = time.time()

xgb = XGBClassifier(n_estimators=100, max_depth=7, learning_rate=0.1, random_state=42, verbosity=0)
xgb.fit(X_train, y_train)
train_time = time.time() - start_time

train_score_xgb = xgb.score(X_train, y_train)
test_score_xgb = xgb.score(X_test, y_test)

print(f"   Training time: {train_time:.2f} seconds")
print(f"   Train accuracy: {train_score_xgb:.4f}")
print(f"   Test accuracy: {test_score_xgb:.4f}")

joblib.dump(xgb, 'models/xgboost.pkl')
print(f"   ✓ Saved model")

# MODEL COMPARISON
print("\n5. MODEL COMPARISON")
print("-" * 70)

models_comparison = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
    'Train Accuracy': [train_score_lr, train_score_rf, train_score_xgb],
    'Test Accuracy': [test_score_lr, test_score_rf, test_score_xgb]
})

print("\n" + models_comparison.to_string(index=False))

# Find best model
best_idx = models_comparison['Test Accuracy'].idxmax()
best_model = models_comparison.loc[best_idx]

print(f"\n🏆 Best Model: {best_model['Model']}")
print(f"   Test Accuracy: {best_model['Test Accuracy']:.4f}")

# FEATURE IMPORTANCE
print("\n6. FEATURE IMPORTANCE")
print("-" * 70)

rf_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 Features (Random Forest):")
for idx, row in rf_importance.head(10).iterrows():
    print(f"   {row['Feature']}: {row['Importance']:.4f}")

# Save results
models_comparison.to_csv('results/model_comparison.csv', index=False)
rf_importance.to_csv('results/feature_importance.csv', index=False)

print("\n" + "="*70)
print("✓ MODEL TRAINING COMPLETE!")
print("="*70)
print(f"\nModels trained and saved:")
print(f"   - models/logistic_regression.pkl")
print(f"   - models/random_forest.pkl")
print(f"   - models/xgboost.pkl")
print(f"\nResults saved:")
print(f"   - results/model_comparison.csv")
print(f"   - results/feature_importance.csv")
print(f"\nNext step: Run 04_model_evaluation.py")
print("="*70)