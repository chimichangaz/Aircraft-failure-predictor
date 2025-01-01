from cleaning import load_aircraft_data
from model import train_random_forest
from featureimportance import plot_feature_importance

# Define base path
base_path = "C:/Users/HP/Documents/Datsets/aircraft"

# Load and process data
train_data, test_data, rul_data = load_aircraft_data(base_path)

# Prepare training data
X_train = train_data.drop(columns=['engine_id', 'cycle', 'RUL'])
y_train = train_data['RUL']

# Prepare test data
X_test = test_data.groupby('engine_id').last().reset_index()  # Last cycle for each engine
X_test = X_test.drop(columns=['engine_id', 'cycle'])
y_test = rul_data['RUL']

# Debug: Verify shapes
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Train the model
rf_model = train_random_forest(X_train, y_train, X_test, y_test)

# Plot feature importance
plot_feature_importance(rf_model, X_train)
