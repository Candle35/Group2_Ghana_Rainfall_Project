# Group 2 — Ghana Rainfall Prediction Using Indigenous Knowledge

## The Business Problem

Across Ghana's Pra River Basin, farmers have no access to modern weather
forecasting tools. For generations they have used indigenous knowledge —
observing the moon, wind, clouds, birds, plants, and stars — to predict
rainfall and plan their farming activities.

These predictions have never been digitised or evaluated systematically.
The Responsible AI Lab (RAIL) at KNUST Ghana developed the SIW Mobile App
to collect structured data from trained farmers who submit their indigenous
weather forecasts. For the first time this traditional knowledge can be
modelled using machine learning.

## The Solution

Build a multi-class classification model that predicts the type of rainfall
expected in the next 12 to 24 hours based solely on indigenous ecological
indicators submitted by farmers in Ghana.

## The Business Value

- Helps rural Ghanaian farmers plan planting, harvesting, and irrigation
- Validates indigenous knowledge using data science
- Provides hyper-local weather predictions where satellite models fail
- Empowers farming communities with AI tools built on their own knowledge
- Contributes to food security across Ghana's Pra River Basin

## The Data

Collected by the Responsible AI Lab (RAIL) at Kwame Nkrumah University of
Science and Technology (KNUST) using the SIW Mobile App. Farmers in the
Pra River Basin submitted indigenous weather forecasts alongside their
ecological indicators. The data was then matched against actual rainfall
measurements from rain gauges.

## Target Variable

| Value | Meaning |
|---|---|
| NORAIN | No rainfall expected |
| SMALLRAIN | Light rainfall expected |
| MEDIUMRAIN | Moderate rainfall expected |
| HEAVYRAIN | Heavy rainfall expected |

Class distribution: NORAIN 87.9%, MEDIUMRAIN 7.0%, HEAVYRAIN 2.9%, SMALLRAIN 2.2%
This is a severely imbalanced multi-class problem.

## Column Descriptions

| Column | Description |
|---|---|
| `ID` | Unique identifier for each forecast submission |
| `user_id` | ID of the farmer who submitted the forecast |
| `confidence` | Farmer's confidence level in their forecast (0.3, 0.6, or 1.0) |
| `predicted_intensity` | Farmer's predicted rainfall intensity (numerical scale) |
| `community` | Name of the farming community where the farmer is located |
| `district` | District in Ghana's Pra River Basin |
| `prediction_time` | Date and time the farmer submitted their forecast |
| `indicator` | Type of indigenous indicator used (clouds, sun, moon, wind, birds, stars, etc.) |
| `indicator_description` | Detailed description of the specific indicator observed |
| `time_observed` | Time the farmer observed the ecological indicator |
| `forecast_length` | How far ahead the forecast is — 12 hours or 24 hours |
| `Target` | Actual rainfall type recorded by rain gauge |

## Key Challenges

- Severe class imbalance: 87.9% of records are NORAIN — model will be biased toward predicting no rain
- indicator and indicator_description have 95%+ missing values — most farmers did not record which indicator they used
- time_observed has 99% missing values — drop this column
- community and district are high cardinality text columns — encode carefully
- Multi-class problem requires different handling than binary classification

## Evaluation Metric

F1 Score — specifically macro F1 which treats all four classes equally.
This penalises the model for ignoring the rare classes (HEAVYRAIN, SMALLRAIN).

## Suggested Approach

1. Drop columns with excessive missing: indicator, indicator_description, time_observed
2. Parse prediction_time to extract hour of day and month
3. Encode community and district using label encoding or target encoding
4. Handle class imbalance using class_weight='balanced' in all models
5. Train and compare: Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM
6. Tune with RandomizedSearchCV scoring='f1_macro'
7. Use classification_report to check F1 per class
