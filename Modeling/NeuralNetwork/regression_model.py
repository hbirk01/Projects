# regression_model_tuned.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your cleaned data
data_cleaned = pd.read_csv('kc_house_data.csv')

# Assuming `y_regression` contains the target variable for regression task (e.g., predicting ProductionCost)
y_regression = data_cleaned['Price']

# Prepare features
X = data_cleaned.drop(columns=['Price'])

# Split the data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y_regression, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Scale the features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Function to create and compile the regression model
def create_model(learning_rate=0.001):
    model = Sequential()
    model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))  # Linear activation for regression

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Hyperparameter tuning grid
learning_rates = [0.001, 0.0005, 0.0001]
batch_sizes = [32, 64, 128]
epochs_list = [10, 20, 30]

# Store the results
tuning_results = []

# Tuning logic: Loop over different configurations of learning rate, batch size, and epochs
for lr in learning_rates:
    for batch_size in batch_sizes:
        for epochs in epochs_list:
            print(f"\nTraining with learning rate={lr}, batch size={batch_size}, epochs={epochs}")
            
            # Create and compile the model
            model = create_model(learning_rate=lr)

            # Train the model
            history = model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=batch_size, 
                                validation_data=(X_val_scaled, y_val), verbose=0)

            # Predict on the test set
            y_test_pred = model.predict(X_test_scaled).ravel()

            # Calculate evaluation metrics (MSE and MAPE)
            mse = mean_squared_error(y_test, y_test_pred)
            mape = mean_absolute_percentage_error(y_test, y_test_pred)

            # Store the results
            tuning_results.append({
                "learning_rate": lr,
                "batch_size": batch_size,
                "epochs": epochs,
                "mse": mse,
                "mape": mape
            })

# Convert results to DataFrame for Seaborn visualization
results_df = pd.DataFrame(tuning_results)

# Plot MSE for different configurations using Seaborn
plt.figure(figsize=(10, 6))
sns.lineplot(x="epochs", y="mse", hue="learning_rate", style="batch_size", data=results_df, markers=True)
plt.title("MSE vs. Epochs for Different Learning Rates and Batch Sizes")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.legend(title="Learning Rate / Batch Size", loc="upper right")
plt.tight_layout()
plt.show()

# Plot MAPE for different configurations using Seaborn
plt.figure(figsize=(10, 6))
sns.lineplot(x="epochs", y="mape", hue="learning_rate", style="batch_size", data=results_df, markers=True)
plt.title("MAPE vs. Epochs for Different Learning Rates and Batch Sizes")
plt.xlabel("Epochs")
plt.ylabel("MAPE")
plt.legend(title="Learning Rate / Batch Size", loc="upper right")
plt.tight_layout()
plt.show()
