import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from xgboost import XGBClassifier
import lightgbm as lgb
from scipy.stats import randint, loguniform
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from src.preprocessing import load_data, prepare_features
from src.config import MODEL_PATH, RANDOM_STATE, TARGET_CLASSES, NUMERICAL_COLS


def get_models_and_params():
    """
    Define all models. Use class_weight='balanced' for all
    to handle the severe class imbalance (87.9% NORAIN).
    """
    models = {
        'Decision Tree': (
            DecisionTreeClassifier(class_weight='balanced', random_state=RANDOM_STATE),
            {
                'max_depth':        randint(3, 15),
                'min_samples_leaf': randint(1, 20)
            }
        ),
        # Add Random Forest with class_weight='balanced'
        'Random Forest': (
            RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1),
            {
                'n_estimators':     randint(50, 200),
                'max_depth':        randint(3, 15),
                'min_samples_leaf': randint(1, 20)
            }
        ),

        # Add XGBoost
        'XGBoost': (
            XGBClassifier(random_state=RANDOM_STATE,eval_metric='mlogloss',verbosity=0),
            {
                'n_estimators':  randint(50, 200),
                'max_depth':     randint(3, 10),
                'learning_rate': loguniform(0.01, 0.3)
            }
        ),

        # Add LightGBM
        'LightGBM': (
            lgb.LGBMClassifier(random_state=RANDOM_STATE,verbosity=-1,class_weight='balanced'),
            {
                'n_estimators':  randint(50, 200),
                'max_depth':     randint(3, 10),
                'learning_rate': loguniform(0.01, 0.3),
                #'num_leaves':    randint(20, 100)
            }
        ),

        # Add Logistic Regression
        'Logistic Regression': (
            Pipeline([
                ('preprocessor', ColumnTransformer(
                    transformers=[
                        ('scaler', StandardScaler(), NUMERICAL_COLS)
                    ],
                    remainder='passthrough'
                )),
                ('model', LogisticRegression(
                    class_weight='balanced',
                    random_state=RANDOM_STATE,
                    max_iter=2000,
                ))
            ]),
            {
                'model__C': loguniform(0.01, 10)
            }
        ),
    }
    return models


def tune_and_compare(models, X_train, y_train, X_test, y_test, n_iter=20, cv=5):
    """
    Run RandomizedSearchCV for each model.
    Use scoring='f1_macro' because this is a multi-class problem.
    """
    results = []
    best_models = {}

    for name, (model, params) in models.items():
        print(f'Tuning {name}...')

        # RandomizedSearchCV with scoring='f1_macro'
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=params,
            n_iter=n_iter,
            scoring='f1_macro',
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0
        )

        # Fit, predict, compute f1_score(y_test, y_pred, average='macro')
        # Fit on training data
        search.fit(X_train, y_train)

        # Best model from search
        best_model = search.best_estimator_

        # Predict on unseen test data
        y_pred = best_model.predict(X_test)

        # Compute macro F1 on test set
        test_f1 = f1_score(y_test, y_pred, average='macro')

        # Print per-class F1 using classification_report
        print(f'[{name}] Best CV F1 Macro : {search.best_score_:.4f}')
        print(f'[{name}] Test F1 Macro    : {test_f1:.4f}')
        print(f'[{name}] Best Params      : {search.best_params_}')
        print(f'\nClassification Report — {name}:')
        print(classification_report(y_test, y_pred, target_names=TARGET_CLASSES))

        # Append results and store best model
        results.append({
            'Model':         name,
            'CV F1 Macro':   round(search.best_score_, 4),
            'Test F1 Macro': round(test_f1, 4),
            'Best Params':   search.best_params_
        })

        best_models[name] = best_model

        pass

    results_df = pd.DataFrame(results).sort_values('Test F1 Macro', ascending=False)
    return results_df, best_models


def save_model(model, path=MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f'Model saved: {path}')


if __name__ == '__main__':
    df = load_data()
    print(f'Data loaded: {df.shape}')

    X_train, X_test, y_train, y_test = prepare_features(df)
    print(f'Train: {X_train.shape}  |  Test: {X_test.shape}')
    print(f'Target classes: {TARGET_CLASSES}')

    models = get_models_and_params()

    print('\nRunning RandomizedSearchCV...\n')
    results_df, best_models = tune_and_compare(models, X_train, y_train, X_test, y_test)

    print('\n--- Model Comparison ---')
    print(results_df[['Model', 'CV F1 Macro', 'Test F1 Macro']].to_string(index=False))

    best_name  = results_df.iloc[0]['Model']
    best_model = best_models[best_name]
    print(f'\nBest model: {best_name}')

    save_model(best_model)
    print('Training complete.')
