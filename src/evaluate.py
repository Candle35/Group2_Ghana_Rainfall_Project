import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score
)
from src.preprocessing import load_data, prepare_features
from src.config import MODEL_PATH, TARGET_CLASSES


def load_model(path=MODEL_PATH):
    with open(path, 'rb') as f:
        return pickle.load(f)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set.
    Print classification report with all four rainfall classes.
    """
    # Generate predictions on unseen test data
    y_pred = model.predict(X_test)

    # Print classification report
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=TARGET_CLASSES))

    # print F1 macro score
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    print(f"Test F1 Macro Score: {macro_f1:.4f}")

    return y_pred


def plot_confusion_matrix(y_test, y_pred):
    """Plot confusion matrix for all four classes.
    - Left:  raw counts (actual number of predictions)
    - Right: normalized (percentage of true label predicted as each class)
    """
    
    # Use TARGET_CLASSES as labels
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=TARGET_CLASSES,
                yticklabels=TARGET_CLASSES)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    df = load_data()
    _, X_test, _, y_test = prepare_features(df)
    model = load_model()
    y_pred = evaluate_model(model, X_test, y_test)
    plot_confusion_matrix(y_test, y_pred)
    plt.show()
