# Customer Churn Prediction System

A machine learning system that predicts which customers are likely to churn using classification algorithms and provides actionable insights to reduce customer attrition.

## Live Demo

**Try it here:** https://churn-prediction-11.streamlit.app/

**View code:** https://github.com/anuja2024/churn-prediction

---

## Project Overview

This project builds an end-to-end machine learning pipeline to predict customer churn in the telecom industry. Using historical customer data from Kaggle, we train multiple models (Logistic Regression, Random Forest, XGBoost) and deploy an interactive Streamlit web app for real-time predictions.

### Dataset: Telco Customer Churn
- **Source:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Customers:** 7,043 customers
- **Features:** 19 customer attributes
- **Target:** Churn (Yes/No)
- **Churn Rate:** 26.5% (imbalanced dataset)

---

## Features

### Home Tab
- Project overview and background
- Dataset statistics and summary
- Model information and architecture
- Quick start guide for new users

### Make Prediction Tab
- Enter customer details (tenure, monthly charges, contract type, etc.)
- Get real-time churn predictions
- View prediction confidence from all 3 models
- Receive actionable insights and recommendations

### Model Performance Tab
- View detailed evaluation metrics for each model
- Compare model accuracy, precision, recall, F1-score, AUC-ROC
- Visualize confusion matrices
- Identify best performing model (XGBoost: 82% accuracy)

### Feature Analysis Tab
- Top 15 most important features affecting churn
- Feature importance visualization
- Understanding which factors drive customer churn
- Model-specific feature rankings

---

## Technologies Used

### Data Processing
- **Python 3.9+** - Programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing

### Machine Learning
- **Scikit-learn** - Machine learning algorithms (Logistic Regression, Random Forest)
- **XGBoost** - Gradient boosting framework
- **Joblib** - Model serialization and persistence

### Visualization
- **Matplotlib & Seaborn** - Static visualizations
- **Plotly** - Interactive visualizations

### Web Application
- **Streamlit** - Web application framework
- **Streamlit Cloud** - Deployment and hosting

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git

### Setup Instructions

#### 1. Clone the Repository
```bash
git clone https://github.com/anuja2024/churn-prediction.git
cd churn-prediction
```

#### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Note:** On Windows, use `--no-build-isolation`:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn plotly streamlit joblib --no-build-isolation
```

#### 4. Download Dataset
1. Go to: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Download the dataset (WA_Fn-UseC_-Telco-Customer-Churn.csv)
3. Create `data/raw/` folder in your project
4. Place the CSV file: `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

---

## Usage

### Option 1: Run the Complete ML Pipeline

Execute scripts in order to process data and train models:

```bash
# 1. Load and explore data
python 01_data_loading.py

# 2. Preprocess and engineer features
python 02_preprocessing.py

# 3. Train all models
python 03_model_training.py

# 4. Evaluate and compare models
python 04_model_evaluation.py
```

**Expected Output:**
- Processed data in `data/processed/`
- Trained models in `models/`
- Evaluation results and plots in `results/`

**Time:** 5-10 minutes total

### Option 2: Run the Streamlit Web App

```bash
streamlit run app.py
```

Then open your browser to: `http://localhost:8501`

**Live Version:** https://churn-prediction-11.streamlit.app/

---

## How to Use the App

### Step 1: Home Tab
- Understand the project
- View dataset statistics
- Learn about the models
- Check model performance summary

### Step 2: Make Prediction Tab
1. Enter customer details:
   - Tenure (months with company)
   - Monthly charges
   - Total charges
   - Contract type
   - Internet service type
   - Online security, tech support, streaming services
2. Click "Predict Churn"
3. View results:
   - Churn probability (0-1 scale)
   - Prediction from each model
   - Risk level (Low/Medium/High)
   - Actionable recommendations

### Step 3: Model Performance Tab
- Compare all 3 models side-by-side
- Understand evaluation metrics:
  - **Accuracy:** Overall prediction correctness
  - **Precision:** Correct positive predictions
  - **Recall:** Actual churners identified
  - **F1-Score:** Balanced metric
  - **AUC-ROC:** Discrimination ability

### Step 4: Feature Analysis Tab
- Identify top factors affecting churn
- Understand feature importance
- See which features matter most for predictions

---

## File Structure

```
churn-prediction/
│
├── 01_data_loading.py              # Load & explore data
├── 02_preprocessing.py             # Clean & engineer features
├── 03_model_training.py            # Train all models
├── 04_model_evaluation.py          # Evaluate & compare models
├── app.py                          # Streamlit web app
│
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .gitignore                      # Git exclusions
│
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Input dataset
│   └── processed/
│       ├── X_train.pkl
│       ├── X_test.pkl
│       ├── y_train.pkl
│       ├── y_test.pkl
│       └── preprocessor.pkl
│
├── models/
│   ├── logistic_regression.pkl     # Baseline model
│   ├── random_forest.pkl           # Ensemble model
│   ├── xgboost.pkl                 # Best model (82% accuracy)
│   └── scaler.pkl                  # Feature scaler
│
└── results/
    ├── model_performance.csv       # Metrics comparison
    ├── feature_importance.csv      # Top features
    ├── confusion_matrix.png        # Visualization
    └── roc_curve.png               # ROC curve plot
```

---

## Model Performance

### Accuracy Results
| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 80% | 64% | 52% | 0.57 | 0.84 |
| Random Forest | 79% | 66% | 50% | 0.57 | 0.83 |
| **XGBoost** | **82%** | **71%** | **68%** | **0.69** | **0.87** |

### Best Model: XGBoost
- **Accuracy:** 82% - Overall correctness
- **Precision:** 71% - When we predict churn, 71% are correct
- **Recall:** 68% - We identify 68% of actual churners
- **F1-Score:** 0.69 - Balanced metric for imbalanced data
- **AUC-ROC:** 0.87 - Excellent discrimination ability

---

## Metrics Explained

### Accuracy
Overall percentage of correct predictions (both churners and non-churners)
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
XGBoost: 82%
```

### Precision
Of customers predicted to churn, how many actually churned
```
Precision = TP / (TP + FP)
XGBoost: 71% - Important to avoid false alarms
```

### Recall
Of customers who actually churned, how many did we identify
```
Recall = TP / (TP + FN)
XGBoost: 68% - Important to catch at-risk customers
```

### F1-Score
Harmonic mean of precision and recall (balanced metric for imbalanced data)
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
XGBoost: 0.69
```

### AUC-ROC
Area under the Receiver Operating Characteristic curve (discrimination ability)
```
Higher is better (0.5 = random, 1.0 = perfect)
XGBoost: 0.87 - Excellent
```

---

## Data Pipeline

### 01_data_loading.py
Loads and explores the Telco customer churn dataset
- Load raw CSV data
- Explore dataset shape, columns, data types
- Check for missing values
- Analyze target variable distribution
- Generate exploratory visualizations

**Output:** Data understanding and EDA plots

### 02_preprocessing.py
Cleans data and engineers features
- Handle missing values
- Encode categorical variables
- Create dummy variables
- Scale numerical features
- Split data (80% train, 20% test)
- Save preprocessed data and scaler

**Output:** 
- `data/processed/X_train.pkl`, `X_test.pkl`
- `data/processed/y_train.pkl`, `y_test.pkl`
- `data/processed/preprocessor.pkl`

### 03_model_training.py
Trains all classification models
- Train Logistic Regression
- Train Random Forest Classifier
- Train XGBoost Classifier
- Hyperparameter tuning
- Cross-validation
- Save trained models as .pkl files

**Output:** 
- `models/logistic_regression.pkl`
- `models/random_forest.pkl`
- `models/xgboost.pkl`
- `models/scaler.pkl`

### 04_model_evaluation.py
Evaluates and compares all models
- Calculate accuracy, precision, recall, F1-score
- Generate confusion matrices
- Plot ROC curves
- Compare model performance
- Identify best model
- Create comparison visualizations

**Output:** 
- `results/model_performance.csv`
- `results/confusion_matrix.png`
- `results/roc_curve.png`

---

## Business Impact

This churn prediction system helps telecom companies:

### Reduce Churn
- Identify at-risk customers before they leave
- Proactively reach out with retention offers
- Understand why customers are churning

### Increase Revenue
- Focus retention efforts on high-value customers
- Reduce customer acquisition costs
- Improve customer lifetime value (CLV)

### Optimize Operations
- Allocate customer service resources efficiently
- Target marketing campaigns effectively
- Improve decision-making with data-driven insights

### Competitive Advantage
- Predict churn faster than competitors
- Implement personalized retention strategies
- Reduce involuntary churn rates

---

## Deployment

### Local Development
```bash
streamlit run app.py
```

### Streamlit Cloud (Live)
1. Push code to GitHub
2. Go to https://share.streamlit.io/
3. Click "New app"
4. Select repository and `app.py`
5. Click "Deploy"

**Live URL:** https://churn-prediction-11.streamlit.app/

### Model Deployment
Models are serialized as `.pkl` files for production:
- **logistic_regression.pkl** - Fast baseline (80% accuracy)
- **random_forest.pkl** - Robust ensemble (79% accuracy)
- **xgboost.pkl** - Best performer (82% accuracy) ⭐

Load in production:
```python
import joblib

model = joblib.load('models/xgboost.pkl')
prediction = model.predict(customer_data)
```

---

## Key Features

### Data Features (19 total)
- **Demographic:** customerID, gender, SeniorCitizen, Partner, Dependents
- **Service:** PhoneService, InternetService, OnlineSecurity, TechSupport, StreamingTV, StreamingMovies, OnlineBackup, DeviceProtection, Contract, PaperlessBilling, PaymentMethod
- **Usage:** Tenure, MonthlyCharges, TotalCharges
- **Target:** Churn (Yes/No)

### Top Churn Drivers (Top 5)
1. **Tenure** - Customers with low tenure are more likely to churn
2. **Monthly Charges** - Higher charges increase churn risk
3. **Contract Type** - Month-to-month contracts have higher churn
4. **Internet Service** - Fiber optic has higher churn than DSL
5. **Online Security** - Lack of online security increases churn

---

## Troubleshooting

### Issue: "Module not found"
```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt --no-build-isolation
```

### Issue: "Data file not found"
- Download from Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- Create `data/raw/` folder
- Place CSV file: `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

### Issue: "Model file not found"
```bash
# Train models first
python 03_model_training.py
```

### Issue: Streamlit app won't start
```bash
# Check requirements
pip list

# Reinstall Streamlit
pip install streamlit --upgrade
```

### Issue: App crashes on Streamlit Cloud
- Verify `requirements.txt` is in root folder
- Ensure `data/` is in `.gitignore`
- Check `app.py` works locally first
- Models should be in `models/` folder

---

## Portfolio Value

This project demonstrates:

### Machine Learning
✓ Binary classification problem (churn prediction)
✓ Multiple model development and comparison
✓ Hyperparameter tuning and optimization
✓ Model evaluation and metrics selection
✓ Handling imbalanced datasets
✓ Feature importance analysis
✓ Cross-validation strategies

### Data Engineering
✓ Data loading and exploration (EDA)
✓ Data cleaning and preprocessing
✓ Feature engineering and scaling
✓ Train/test splitting
✓ Data visualization
✓ Statistical analysis

### Web Development
✓ Streamlit framework expertise
✓ Interactive web app design
✓ Real-time predictions
✓ Multi-tab interface
✓ Data visualization (Plotly, Matplotlib)
✓ Cloud deployment

### Business Analytics
✓ Understanding churn drivers
✓ Feature importance interpretation
✓ ROI and business impact calculation
✓ Actionable insights generation
✓ Decision-making support

### Software Engineering
✓ Modular code organization
✓ Production-ready code quality
✓ Model serialization and deployment
✓ Complete documentation
✓ Version control (Git/GitHub)

---

## Use Cases

### Data Scientist Role
```
Customer Churn Prediction
- Built 3 ML models (Logistic Regression, Random Forest, XGBoost)
- Achieved 82% accuracy with XGBoost
- Created end-to-end data pipeline
- Deployed interactive web app
- Live demo: https://churn-prediction-11.streamlit.app/
```

### Analytics Role
```
Customer Churn Prediction
- Identified top 5 churn drivers
- Created interactive dashboards
- Analyzed model performance metrics
- Generated retention strategies
- Calculated business impact
- Live demo: https://churn-prediction-11.streamlit.app/
```

### Full-Stack Role
```
Customer Churn Prediction
- Data pipeline (loading → preprocessing → training)
- ML model training and evaluation
- Web app development (Streamlit)
- Cloud deployment (Streamlit Cloud)
- Live demo: https://churn-prediction-11.streamlit.app/
```

---

## Future Improvements

- Add SHAP values for model interpretability
- Implement automated retraining pipeline
- Add A/B testing framework
- Integrate with real customer database
- Deploy on cloud platform (AWS/GCP/Azure)
- Add email notification system for at-risk customers
- Create API endpoint for predictions
- Implement feature monitoring and drift detection
- Add customer segmentation analysis
- Build recommendation engine for retention offers

---

## Requirements

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.0
xgboost>=1.5.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0
streamlit>=1.0.0
joblib>=1.1.0
```

---

## License

Educational project using publicly available datasets from Kaggle.

---

## Author

**Anuja Patade**
- Email: anujapatade2003@gmail.com
- GitHub: https://github.com/anuja2024
- Portfolio: [Your portfolio URL]

---

## Acknowledgments

- Kaggle for the Telco Customer Churn dataset
- Scikit-learn for machine learning algorithms
- XGBoost for gradient boosting
- Streamlit for the web framework
- Plotly for interactive visualizations

---

## Quick Links

- **Live Demo:** https://churn-prediction-11.streamlit.app/
- **GitHub Repository:** https://github.com/anuja2024/churn-prediction
- **Dataset:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Scikit-learn:** https://scikit-learn.org/
- **XGBoost:** https://xgboost.readthedocs.io/
- **Streamlit:** https://docs.streamlit.io/
- **Plotly:** https://plotly.com/python/

---

## Contact & Support

For questions or issues:
1. Check the TROUBLESHOOTING section above
2. Review GitHub issues
3. Contact: anujapatade2003@gmail.com

---

**Project Status:** Production Ready ✅

**Created:** June 2026

**Python Version:** 3.8+

**Framework:** Streamlit 1.0+

**Best Model:** XGBoost (82% accuracy)

---

**Ready to predict customer churn and improve retention?**

**Try the live demo:** https://churn-prediction-11.streamlit.app/ 🚀
