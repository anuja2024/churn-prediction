"""
CHURN PREDICTION - STREAMLIT WEB APP
Interactive app to predict customer churn
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

# Page config
st.set_page_config(
    page_title="Churn Prediction App",
    page_icon="chart",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("Telecom Churn Prediction")
st.markdown("Predict which customers are likely to leave")

# ============================================================
# LOAD MODELS AND DATA
# ============================================================

@st.cache_resource
def load_models():
    """Load trained models"""
    lr = joblib.load('models/logistic_regression.pkl')
    rf = joblib.load('models/random_forest.pkl')
    xgb = joblib.load('models/xgboost.pkl')
    scaler = joblib.load('models/scaler.pkl')
    label_encoders = joblib.load('models/label_encoders.pkl')
    
    return lr, rf, xgb, scaler, label_encoders

@st.cache_data
def load_data():
    """Load evaluation metrics"""
    metrics_df = pd.read_csv('results/evaluation_metrics.csv')
    feature_importance = pd.read_csv('results/feature_importance.csv')
    return metrics_df, feature_importance

# Load everything
lr, rf, xgb, scaler, label_encoders = load_models()
metrics_df, feature_importance = load_data()

models = {
    'Logistic Regression': lr,
    'Random Forest': rf,
    'XGBoost': xgb
}

# ============================================================
# TABS - HOME AND PREDICTION
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(["Home", "Make Prediction", "Model Performance", "Feature Analysis"])

# ============================================================
# TAB 1: HOME
# ============================================================

with tab1:
    st.header("Welcome to Churn Prediction System")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Total Customers", value="7,043")
    
    with col2:
        st.metric(label="Churn Rate", value="26.5%")
    
    with col3:
        st.metric(label="Models", value="3")
    
    # st.markdown("""
    # ### What is Churn?
    # Churn happens when a customer leaves your service.
    
    # ### Why is it important?
    # - It's cheaper to keep customers than find new ones
    # - Predicting churn helps you take action before they leave
    # - You can offer special deals or better service
    
    # ### How does this app work?
    # 1. Input customer data - Tell us about the customer
    # 2. Model predicts - Our AI predicts if they'll churn
    # 3. See results - Get a prediction and explanation
    
    # ### Models Used
    # - Logistic Regression - Simple, fast baseline
    # - Random Forest - Powerful ensemble method
    # - XGBoost - State-of-the-art gradient boosting
    
    # ### Dataset Information
    # - Source: Kaggle Telecom Customer Churn
    # - Customers: 7,043
    # - Features: 20 customer attributes
    # - Churn Rate: 26.5% (1,869 customers left)
    
    # ### Get Started
    # Go to the "Make Prediction" tab to predict churn for a customer!
    # """)

# ============================================================
# TAB 2: MAKE PREDICTION
# ============================================================

with tab2:
    st.header("Make a Prediction")
    
    st.subheader("Enter Customer Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Personal Information")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen?", ["No", "Yes"])
        partner = st.selectbox("Has Partner?", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 30)
        phone_service = st.selectbox("Phone Service?", ["No", "Yes"])
    
    with col2:
        st.write("Service Information")
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security?", ["No", "Yes"])
        tech_support = st.selectbox("Tech Support?", ["No", "Yes"])
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 119.0, 65.0)
        total_charges = st.slider("Total Charges ($)", 20.0, 8684.0, 2000.0)
    
    st.markdown("---")
    
    # Create prediction input
    if st.button("Predict Churn", use_container_width=True):
        
        # Prepare data with ALL required columns
        input_data = {
            'gender': gender,
            'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': 'No',
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': 'No',
            'DeviceProtection': 'No',
            'TechSupport': tech_support,
            'StreamingTV': 'No',
            'StreamingMovies': 'No',
            'Contract': contract,
            'PaperlessBilling': 'No',
            'PaymentMethod': 'Electronic check',
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }
        
       # Create dataframe
        df_input = pd.DataFrame([input_data])
        
        # Encode ALL categorical variables
        categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                           'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                           'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                           'PaperlessBilling', 'PaymentMethod']
        
        for col in categorical_cols:
            if col in label_encoders:
                df_input[col] = label_encoders[col].transform(df_input[col].astype(str))
        
        # Scale
        df_input_scaled = scaler.transform(df_input)
        
        # Make predictions
        st.subheader("Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pred_lr = lr.predict_proba(df_input_scaled)[0][1]
            st.metric("Logistic Regression", f"{pred_lr*100:.1f}%")
        
        with col2:
            pred_rf = rf.predict_proba(df_input_scaled)[0][1]
            st.metric("Random Forest", f"{pred_rf*100:.1f}%")
        
        with col3:
            pred_xgb = xgb.predict_proba(df_input_scaled)[0][1]
            st.metric("XGBoost", f"{pred_xgb*100:.1f}%")
        
        # Average prediction
        avg_pred = (pred_lr + pred_rf + pred_xgb) / 3
        
        st.markdown("---")
        
        if avg_pred > 0.5:
            st.error(f"HIGH CHURN RISK: {avg_pred*100:.1f}%")
            st.write("This customer is likely to churn. Consider offering special retention deals!")
        else:
            st.success(f"LOW CHURN RISK: {avg_pred*100:.1f}%")
            st.write("This customer is likely to stay. Continue good service!")

# ============================================================
# TAB 3: MODEL PERFORMANCE
# ============================================================

with tab3:
    st.header("Model Performance Comparison")
    
    # Metrics table
    st.subheader("Evaluation Metrics")
    st.dataframe(metrics_df.set_index('Model'), use_container_width=True)
    
    # Accuracy comparison
    st.subheader("Accuracy Comparison")
    fig = px.bar(
        metrics_df,
        x='Model',
        y='Accuracy',
        color='Model',
        title="Model Accuracy on Test Set",
        labels={'Accuracy': 'Accuracy Score'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # All metrics comparison
    st.subheader("All Metrics Comparison")
    metrics_for_plot = metrics_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']]
    metrics_melted = metrics_for_plot.melt(id_vars=['Model'], var_name='Metric', value_name='Score')
    
    fig = px.bar(
        metrics_melted,
        x='Metric',
        y='Score',
        color='Model',
        barmode='group',
        title="Model Metrics Comparison"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Key insights
    st.subheader("Key Insights")
    best_accuracy = metrics_df.loc[metrics_df['Accuracy'].idxmax()]
    best_f1 = metrics_df.loc[metrics_df['F1-Score'].idxmax()]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"Best Accuracy: {best_accuracy['Model']} ({best_accuracy['Accuracy']:.4f})")
    
    with col2:
        st.info(f"Best F1-Score: {best_f1['Model']} ({best_f1['F1-Score']:.4f})")

# ============================================================
# TAB 4: FEATURE ANALYSIS
# ============================================================

with tab4:
    st.header("Feature Importance Analysis")
    
    st.subheader("Top 15 Most Important Features")
    
    top_features = feature_importance.head(15)
    
    fig = px.bar(
        top_features,
        x='Importance',
        y='Feature',
        orientation='h',
        title="Feature Importance (Random Forest)",
        labels={'Importance': 'Importance Score', 'Feature': 'Feature Name'}
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("What these features mean:")
    
    st.markdown("""
    Top Features for Churn Prediction:
    
    - Tenure - How long customer has been with company (longer = less likely to churn)
    - Monthly Charges - How much they pay per month
    - Contract - Type of contract (month-to-month = higher churn)
    - Internet Service - Type of internet (Fiber optic = higher churn)
    - Tech Support - Whether they have tech support
    
    Key Insight: Longer tenure is the strongest predictor of retention!
    """)

# ============================================================
# FOOTER
# ============================================================

# st.markdown("---")
# st.markdown("""
# <div style='text-align: center'>
#     <p>Telecom Churn Prediction App | Built with Streamlit | ML Models: LR, RF, XGBoost</p>
#     <p style='font-size: 12px; color: gray;'>Churn Prediction System for Data Science Portfolio</p>
# </div>
# """, unsafe_allow_html=True)