# classification_model_tuned.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the cleaned dataset
data_cleaned = pd.read_csv('cleaned_manufacturing_data.csv')

# Prepare features and target
X = data_cleaned.drop(columns=['DefectStatus'])  # Features
y = data_cleaned['DefectStatus']  # Target

# Print dataset size
dataset_size = X.shape[0]
print(f"Dataset size: {dataset_size} samples")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Scale the features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Function to create and compile the model
def create_model(learning_rate=0.001):
    model = Sequential()
    model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification output layer

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Define hyperparameter tuning grid
learning_rates = [0.001, 0.0005, 0.0001]
batch_sizes = [32, 64, 128]
epochs_list = [10, 20, 30]

# Results storage for tuning
tuning_results = []

# Tuning logic (looping through learning rates, batch sizes, and epochs)
for lr in learning_rates:
    for batch_size in batch_sizes:
        for epochs in epochs_list:
            print(f"\nTraining with learning rate={lr}, batch size={batch_size}, epochs={epochs}")
            
            # Create and compile the model
            model = create_model(learning_rate=lr)

            # Train the model and save the history
            history = model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=batch_size, 
                                validation_data=(X_test_scaled, y_test), verbose=0)

            # Evaluate the model
            test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
            print(f"Test Accuracy: {test_acc:.4f}")

            # Store results
            tuning_results.append({
                "learning_rate": lr,
                "batch_size": batch_size,
                "epochs": epochs,
                "accuracy": test_acc,
                "loss": test_loss
            })

# Convert results to DataFrame for better visualization
results_df = pd.DataFrame(tuning_results)

# Plot the tuning results with Seaborn (Accuracy)
plt.figure(figsize=(10, 6))
sns.lineplot(x="epochs", y="accuracy", hue="learning_rate", style="batch_size", data=results_df, markers=True)
plt.title("Accuracy vs. Epochs for Different Learning Rates and Batch Sizes")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(title="Learning Rate / Batch Size", loc="lower right")
plt.tight_layout()
plt.show()

# Plot the tuning results with Seaborn (Loss)
plt.figure(figsize=(10, 6))
sns.lineplot(x="epochs", y="loss", hue="learning_rate", style="batch_size", data=results_df, markers=True)
plt.title("Loss vs. Epochs for Different Learning Rates and Batch Sizes")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(title="Learning Rate / Batch Size", loc="upper right")
plt.tight_layout()
plt.show()
