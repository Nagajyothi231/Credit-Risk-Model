# Credit Risk Model

## Project Overview
This project is focused on building an effective credit risk model to optimize lending decisions for credit card companies, such as American Express (Amex). By leveraging data-driven insights, this model aims to enhance the customer experience while ensuring a sound risk strategy that maximizes revenue. Two advanced machine learning models were implemented and compared: **XGBoost** and a **Neural Network**.

## Objectives
- Develop aggressive and conservative credit risk strategies.
- Optimize lending decisions by identifying high-risk and low-risk customers.
- Enhance model accuracy and interpretability to support data-driven decision-making.

## Data
The dataset includes 13 months of historical data with 91,783 unique customer records. Key data categories include:
- **Delinquency**
- **Balance**
- **Risk**
- **Spend**
- **Payment**

**Data Source:** [https://www.kaggle.com/competitions/amex-default-prediction/data](#)  

## Methodology

### Feature Engineering
- Aggregate functions (mean, min, max, last values of 12 months) were used for numerical features.
- One-hot encoding of 11 categorical features resulted in 45 new features.
- Important features were selected using XGBoost feature importance scores (> 0.5%).

### Model Development
#### **1. XGBoost**
- **Grid Search**:
  - Parameters: Number of trees, learning rate, percentage of observations and features, weight of default observations.
  - 72 models were trained to identify the best configuration.
- **SHAP Analysis**:
  - P_2_last was identified as the most impactful feature, with positive contributions observed for several others.

#### **2. Neural Network**
- **Data Processing**:
  - Outliers modified to the 99th percentile.
  - Data normalized using the standard scaler function.
  - Missing values replaced with zeros.
- **Grid Search**:
  - Configurations included 2 and 4 hidden layers, ReLU and Tanh activation functions, dropout regularization, and varying batch sizes.
  - 32 models were trained to optimize AUC scores.

## Results
- The XGBoost model consistently achieved higher AUC scores compared to the Neural Network, making it the preferred choice for this dataset.

## Key Insights
- Effective feature selection and hyperparameter tuning significantly enhance model performance.
- SHAP analysis provides valuable insights into feature impact, aiding interpretability.

## Technologies Used
- **Python**: Data processing, model training, and evaluation.
- **Libraries**: XGBoost, TensorFlow, Scikit-learn, Pandas, NumPy, Matplotlib.

## Contributors
This project was collaboratively developed by:
- Naga Jyothi Muvva

## Contact
For questions or further information, please feel free to reach out via the project repository's Issues section.
