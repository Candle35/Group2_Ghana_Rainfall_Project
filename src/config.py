import os

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, 'data', 'rainfall.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.pkl')

TARGET_COL   = 'Target'
TEST_SIZE    = 0.2
RANDOM_STATE = 42

# These columns have 95%+ missing values — drop them
DROP_COLS = ['ID', 'indicator', 'indicator_description', 'time_observed']

CATEGORICAL_COLS = ['community', 'district']

NUMERICAL_COLS = [
    'user_id', 'confidence', 'predicted_intensity', 'forecast_length',
    'prediction_hour', 'prediction_month'
]

# Target class labels
TARGET_CLASSES = ['NORAIN', 'SMALLRAIN', 'MEDIUMRAIN', 'HEAVYRAIN']

MODEL_PARAMS = {
    'n_estimators':  200,
    'learning_rate': 0.05,
    'num_leaves':    31,
    'random_state':  RANDOM_STATE,
    'verbose':       -1,
    'class_weight':  'balanced'
}
