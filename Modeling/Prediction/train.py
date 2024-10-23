import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from joblib import dump

def remove_outliers(df, features, multiplier=1.5):
    for feature in features:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]
    return df

# Data loading and preprocessing
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)

    features_to_clean = ["ProductionVolume", "ProductionCost", "DeliveryDelay", "DefectRate",
                         "QualityScore", "MaintenanceHours", "DowntimePercentage", 
                         "InventoryTurnover", "StockoutRate", "WorkerProductivity", 
                         "EnergyConsumption", "EnergyEfficiency", "AdditiveProcessTime", "AdditiveMaterialCost"]

    # Outlier removal, missing value handling, and scaling
    data_cleaned = remove_outliers(data, features_to_clean)
    imputer = SimpleImputer(strategy='median')
    data_cleaned[features_to_clean] = imputer.fit_transform(data_cleaned[features_to_clean])
    scaler = MinMaxScaler()
    data_cleaned[features_to_clean] = scaler.fit_transform(data_cleaned[features_to_clean])

    return data_cleaned

# Training the model
def train_model(data_file):
    data = load_and_preprocess_data(data_file)
    X = data.drop(columns=['DefectStatus'])
    y = data['DefectStatus']

    # SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.25, random_state=42)

    # RandomForest model training
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the model
    dump(model, 'random_forest_defect_model.joblib')

if __name__ == "__main__":
    train_model("manufacturing_defect_dataset.csv")
