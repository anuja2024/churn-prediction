"""
CHURN PREDICTION - STEP 4: MODEL EVALUATION
Detailed evaluation with metrics and visualizations
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, classification_report
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("CHURN PREDICTION - MODEL EVALUATION")
print("="*70)

# Load data and models
print("\n1. LOADING DATA AND MODELS...")
X_test = pd.read_csv('data/processed/X_test.csv')
y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()

lr = joblib.load('models/logistic_regression.pkl')
rf = joblib.load('models/random_forest.pkl')
xgb = joblib.load('models/xgboost.pkl')

models = {
    'Logistic Regression': lr,
    'Random Forest': rf,
    'XGBoost': xgb
}

print(f"   ✓ Loaded test data: {X_test.shape}")
print(f"   ✓ Loaded 3 models")

# Calculate metrics
print("\n2. CALCULATING METRICS...")
print("-" * 70)

results = []

for name, model in models.items():
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC-ROC': auc_roc,
        'Model_Object': model,
        'Predictions': y_pred,
        'Probabilities': y_pred_proba
    })
    
    print(f"\n{name}:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   AUC-ROC:   {auc_roc:.4f}")

# Save results
results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'Model_Object'} 
                           for r in results])
results_df.to_csv('results/evaluation_metrics.csv', index=False)
print(f"\n   ✓ Saved to results/evaluation_metrics.csv")

# Classification reports
print("\n3. CLASSIFICATION REPORTS")
print("-" * 70)

for result in results:
    model = result['Model_Object']
    y_pred = result['Predictions']
    name = result['Model']
    
    print(f"\n{name}:")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

# Confusion matrices
print("\n4. CREATING CONFUSION MATRICES...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Confusion Matrices', fontsize=16, fontweight='bold')

for idx, result in enumerate(results):
    model = result['Model_Object']
    y_pred = result['Predictions']
    name = result['Model']
    
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                cbar=False, square=True)
    axes[idx].set_title(f'{name}', fontweight='bold')
    axes[idx].set_ylabel('Actual')
    axes[idx].set_xlabel('Predicted')

plt.tight_layout()
plt.savefig('results/confusion_matrices.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved to results/confusion_matrices.png")
plt.close()

# ROC curves
print("\n5. CREATING ROC CURVES...")

plt.figure(figsize=(10, 8))

for result in results:
    model = result['Model_Object']
    y_pred_proba = result['Probabilities']
    name = result['Model']
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)

plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
plt.legend(fontsize=11, loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/roc_curves.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved to results/roc_curves.png")
plt.close()

# Model comparison chart
print("\n6. CREATING MODEL COMPARISON CHART...")

metrics_df = results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']].copy()

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Model Metrics Comparison', fontsize=16, fontweight='bold')

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
colors = ['#3498db', '#e74c3c', '#2ecc71']

for idx, metric in enumerate(metrics):
    ax = axes[idx // 3, idx % 3]
    
    values = metrics_df[metric].values
    ax.bar(metrics_df['Model'], values, color=colors)
    ax.set_ylabel(metric, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.set_xticklabels(metrics_df['Model'], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

axes[1, 2].remove()

plt.tight_layout()
plt.savefig('results/model_comparison_chart.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved to results/model_comparison_chart.png")
plt.close()

# Summary
print("\n" + "="*70)
print("✓ MODEL EVALUATION COMPLETE!")
print("="*70)

print(f"\n📊 Model Performance Summary:")
print(metrics_df.to_string(index=False))

# Find best model
best_accuracy = metrics_df.loc[metrics_df['Accuracy'].idxmax()]
best_f1 = metrics_df.loc[metrics_df['F1-Score'].idxmax()]
best_auc = metrics_df.loc[metrics_df['AUC-ROC'].idxmax()]

print(f"\n🏆 Best Models:")
print(f"   Best Accuracy: {best_accuracy['Model']} ({best_accuracy['Accuracy']:.4f})")
print(f"   Best F1-Score: {best_f1['Model']} ({best_f1['F1-Score']:.4f})")
print(f"   Best AUC-ROC: {best_auc['Model']} ({best_auc['AUC-ROC']:.4f})")

print(f"\n📁 Visualizations created:")
print(f"   - results/confusion_matrices.png")
print(f"   - results/roc_curves.png")
print(f"   - results/model_comparison_chart.png")

print(f"\n✓ PROJECT COMPLETE!")
print("="*70)