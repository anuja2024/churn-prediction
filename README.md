## Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Setup

1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/churn-prediction.git
cd churn-prediction
```

2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Download dataset
- Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- Extract and place in: `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

## Usage

### Run the ML Pipeline
```bash
python 01_data_loading.py
python 02_preprocessing.py
python 03_model_training.py
python 04_model_evaluation.py
```

### Run the Streamlit App
```bash
streamlit run app.py
```

Then open your browser to: `http://localhost:8501`

## App Features

### 1. Home Tab
- Project overview
- Dataset statistics
- Model information
- Quick start guide

### 2. Make Prediction Tab
- Enter customer details
- Get real-time predictions
- View prediction confidence from all 3 models
- Receive actionable insights

### 3. Model Performance Tab
- View detailed evaluation metrics
- Compare model accuracy
- Analyze precision, recall, F1-score, and AUC-ROC
- Identify best performing model

### 4. Feature Analysis Tab
- Top 15 most important features
- Feature importance visualization
- Understanding feature impact on churn

## Technologies Used
- **Python 3.9+** - Programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning algorithms
- **XGBoost** - Gradient boosting framework
- **Matplotlib & Seaborn** - Static visualizations
- **Plotly** - Interactive visualizations
- **Streamlit** - Web application framework
- **Joblib** - Model serialization

## Business Impact
This model can help telecom companies:
- Identify at-risk customers before they leave
- Implement targeted retention strategies
- Reduce customer acquisition costs
- Improve customer lifetime value
- Optimize marketing spend

## Model Deployment
The models are trained and serialized as `.pkl` files for production use:
- `logistic_regression.pkl` - Fast, interpretable baseline
- `random_forest.pkl` - Ensemble learning approach
- `xgboost.pkl` - State-of-the-art gradient boosting

## Performance Metrics Explained

**Accuracy**: Overall correctness of predictions
- Higher is better
- XGBoost: 82%

**Precision**: Of predicted churners, how many actually churned
- Important to avoid false alarms
- XGBoost: 71%

**Recall**: Of actual churners, how many we identified
- Important to catch at-risk customers
- XGBoost: 68%

**F1-Score**: Harmonic mean of precision and recall
- Balanced metric for imbalanced data
- XGBoost: 0.69

**AUC-ROC**: Area under the ROC curve
- Measures model discrimination ability
- XGBoost: 0.87

## Future Improvements
- Add SHAP values for model interpretability
- Implement automated retraining pipeline
- Add A/B testing framework
- Integrate with real customer database
- Deploy on cloud platform (AWS/GCP/Azure)
- Add email notification system for at-risk customers

## Author
Anuja Patade
