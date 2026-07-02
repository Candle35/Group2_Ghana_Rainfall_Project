import pickle
import pandas as pd
from src.preprocessing import clean_data, encode_features
from src.config import MODEL_PATH, TARGET_COL, TARGET_CLASSES


def load_model(path=MODEL_PATH):
    with open(path, 'rb') as f:
        return pickle.load(f)


def align_columns(df, model):
    """
    Ensure the input DataFrame has the same columns the model was trained on,
    in the same order. Drops any extra columns the model does not expect.
    """
    if hasattr(model, 'feature_names_in_'):
        cols = [c for c in model.feature_names_in_ if c in df.columns]
        return df[cols]
    return df


def predict_single(model, input_dict):
    """
    Predict rainfall type for a single farmer submission.
    Returns dict with prediction (int), label (NORAIN/SMALLRAIN/etc), probabilities.
    """
    # DataFrame, clean, encode, align, predict
    # Step 1 — convert the single input dictionary into a one-row DataFrame
    df = pd.DataFrame([input_dict])

    # Step 2 — clean and encode exactly as done during training
    df = clean_data(df)
    df = encode_features(df)

    # Step 3 — align columns to match what the model was trained on
    df = align_columns(df, model)

    # Step 4 — predict the class integer (0, 1, 2, or 3)
    prediction = int(model.predict(df)[0])

    # Step 5 — get class probabilities if the model supports it
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(df)[0]
        probabilities = {cls: round(float(p), 4) for cls, p in zip(TARGET_CLASSES, proba)}
    else:
        probabilities = {}

    # return {'prediction': int, 'label': TARGET_CLASSES[prediction],
    #               'probabilities': dict of class probabilities}
    return {
        'prediction': prediction,
        'label': TARGET_CLASSES[prediction],
        'probabilities': probabilities
    }


def predict_batch(model, df):
    """
    Predict rainfall type for a batch of submissions.
    Returns df with Prediction, Label columns added.
    """
    # copy, clean, encode, align, predict
    # Step 1 — copy to avoid modifying the original DataFrame
    df = df.copy()

    # Step 2 — clean and encode exactly as done during training
    df = clean_data(df)
    df = encode_features(df)

    # Step 3 — align columns to match what the model was trained on
    df = align_columns(df, model)

    # Step 4 — predict class integers for all rows
    predictions = model.predict(df)

    # map prediction integers back to class names using TARGET_CLASSES
    df['Prediction'] = predictions
    df['Label'] = [TARGET_CLASSES[p] for p in predictions]  # mapping happens directly
    print(f"[predict_batch] Predicted {len(df):,} submissions.")
    print(f"[predict_batch] Label distribution:\n{df['Label'].value_counts()}")
    return df
