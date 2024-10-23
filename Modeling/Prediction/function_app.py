import azure.functions as func
import pandas as pd
from joblib import load
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import logging

# Define the Azure Function app instance
app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# Load the trained model and selected features
model = load('random_forest_defect_model.joblib')

def preprocess_input(data):
    # List of all features used during model training
    expected_features = [
        "ProductionVolume", "ProductionCost", "SupplierQuality", "DeliveryDelay", "DefectRate", 
        "QualityScore", "MaintenanceHours", "DowntimePercentage", "InventoryTurnover", 
        "StockoutRate", "WorkerProductivity", "SafetyIncidents", "EnergyConsumption", 
        "EnergyEfficiency", "AdditiveProcessTime", "AdditiveMaterialCost"
    ]
    
    # Ensure all expected columns are present
    missing_features = [feature for feature in expected_features if feature not in data.columns]
    
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Proceed with preprocessing steps as in training
    imputer = SimpleImputer(strategy='median')
    data[expected_features] = imputer.fit_transform(data[expected_features])

    scaler = MinMaxScaler()
    data[expected_features] = scaler.fit_transform(data[expected_features])

    return data

# Define the HTTP-triggered function
@app.route(route="defect_analysis")
def defect_analysis(req: func.HttpRequest) -> func.HttpResponse:
    try:
        req_body = req.get_json()
        df = pd.DataFrame([req_body])

        # Check for actual defect status
        actual_defect_status = req_body.get("DefectStatus", "Not provided")

        # Preprocess input data
        df_cleaned = preprocess_input(df)

        # Get the prediction
        prediction = model.predict(df_cleaned)

        return func.HttpResponse(
            f"Predicted Defect Status: {prediction[0]}. Actual Defect Status (if provided): {actual_defect_status}", 
            status_code=200
        )

    except Exception as e:
        return func.HttpResponse(f"Error: {str(e)}", status_code=500)

