import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from src.config import (
    DATA_PATH, TARGET_COL, DROP_COLS,
    CATEGORICAL_COLS, TARGET_CLASSES, TEST_SIZE, RANDOM_STATE
)


def load_data(path=DATA_PATH):
    """Load the Ghana rainfall dataset."""
    # Read the CSV file and return a DataFrame
    df = pd.read_csv(path)
    print(f"[load_data] Loaded {df.shape[0]:,} rows x {df.shape[1]} columns.")
    return df

def clean_data(df):
    """
    Clean the dataset.
    Steps:
    1. Drop columns in DROP_COLS (high missing value columns)
    2. Parse prediction_time and extract hour and month
    3. Normalize community names (strip whitespace, collapse spaces, title case)
    4. Map known community names to their original names
    """
    df = df.copy()

    # Step 1 — drop columns in DROP_COLS
    df = df.drop(columns=[col for col in DROP_COLS if col in df.columns])

    # Step 2 — parse prediction_time and extract features
    #df['prediction_time'] = pd.to_datetime(df['prediction_time'], errors='coerce')
    #df['prediction_hour']  = df['prediction_time'].dt.hour
    #df['prediction_month'] = df['prediction_time'].dt.month
    #df = df.drop(columns=['prediction_time'])

    if 'prediction_time' in df.columns:
        df['prediction_time'] = pd.to_datetime(df['prediction_time'], errors='coerce')
        df['prediction_hour'] = df['prediction_time'].dt.hour
        df['prediction_month'] = df['prediction_time'].dt.month
        df = df.drop(columns=['prediction_time'])


    # Create time features only if prediction_time exists
    return df

def normalize_text_column(df, col):
    """
    Normalize community column by:
    1. Stripping leading/trailing whitespace
    2. Collapsing multiple internal spaces into one
    3. Converting to title case for consistency
    """
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.replace(r'\s+', ' ', regex=True)
            .str.title()
        )
    return df

# Manual mapping of known community aliases to their canonical names
COMMUNITY_ALIASES = {
    'Foso Odumasi': 'Assin Foso Odumasi',
    'Atonsu': 'Assin Atonsu',
    'Odumasi Adansi': 'Odumasi'
}

def resolve_community_aliases(df, col='community'):
    """
    Map known communities to their original names.
    Must be called AFTER normalize_text_column so cases and spaces are already clean.
    """
    if col in df.columns:
        df[col] = df[col].replace(COMMUNITY_ALIASES)
    return df

def encode_features(df):
    """
    Encode categorical columns (community, district) using LabelEncoder.
    """
    df = df.copy()
    le = LabelEncoder()

    # Loop through CATEGORICAL_COLS and encode each one
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
    return df


def prepare_features(df):
    """
    Full pipeline — returns X_train, X_test, y_train, y_test.
    Encode target: NORAIN=0, SMALLRAIN=1, MEDIUMRAIN=2, HEAVYRAIN=3
    """
    # Step 1 — clean
    df = clean_data(df)

    # Step 2 — encode target
    target_map = {c: i for i, c in enumerate(TARGET_CLASSES)}
    df[TARGET_COL] = df[TARGET_COL].map(target_map)

    # Step 3 — encode features
    df = encode_features(df)

    # Step 4 — split X and y, then train_test_split with stratify=y
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    return X_train, X_test, y_train, y_test
