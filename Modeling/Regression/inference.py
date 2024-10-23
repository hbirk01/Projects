# inference.py
import pandas as pd
import joblib

# Load the saved model
model = joblib.load('best_defect_model.pkl')

def predict_defects(input_data):
    """Input data must be a DataFrame with the same structure as the training data."""
    return model.predict(input_data)
