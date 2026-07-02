# Group 2 — Ghana Rainfall Predictor

## Setup
```bash
conda create -n ghana_rain python=3.10
conda activate ghana_rain
pip install -r requirements.txt
```

## Data
Place `rainfall.csv` in the `data/` folder.

## Run
```bash
python src/train.py
python src/evaluate.py
streamlit run app.py
```

## Files to Complete
| File | What to do |
|---|---|
| src/preprocessing.py | Complete all TODO sections |
| src/train.py | Add all 5 models and complete tune_and_compare() |
| src/evaluate.py | Complete evaluate_model() and plot functions |
| src/predict.py | Complete predict_single() and predict_batch() |
