"""
Configuration settings for Omnistal v1.5
"""

from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

# File paths
RAW_DATA_FILE = RAW_DATA_DIR / "ibm_hr.csv"
DB_FILE = PROCESSED_DATA_DIR / "omnistal.db"
MODEL_FILE = MODELS_DIR / "attrition_model.pkl"

# Model configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Feature configuration
FEATURE_COLUMNS = [
    'Age',
    'MonthlyIncome',
    'YearsAtCompany',
    'YearsInCurrentRole',
    'YearsSinceLastPromotion',
    'YearsWithCurrManager',
    'NumCompaniesWorked',
    'DistanceFromHome',
    'PercentSalaryHike',
    'TrainingTimesLastYear',
    'EnvironmentSatisfaction',
    'JobSatisfaction',
    'WorkLifeBalance',
    'JobInvolvement',
    'PerformanceRating',
    'OverTime'
]

# Categorical columns to encode
CATEGORICAL_COLUMNS = [
    'OverTime',
    'Gender',
    'MaritalStatus',
    'Department',
    'JobRole',
    'BusinessTravel'
]

# Risk thresholds
RISK_THRESHOLD_HIGH = 70
RISK_THRESHOLD_MEDIUM = 40

# Model hyperparameters
XGBOOST_PARAMS = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'objective': 'binary:logistic',
    'random_state': RANDOM_STATE,
    'eval_metric': 'auc'
}

# Database configuration
DB_TABLE_NAME = 'employees'
DB_PREDICTIONS_TABLE = 'predictions'

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
