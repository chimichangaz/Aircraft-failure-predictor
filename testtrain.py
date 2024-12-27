# Load the processed data
from cleaning import load_aircraft_data  # Ensure your data loader function is saved in a module
base_path = "C:/Users/HP/Documents/Datsets/aircraft"
train_combined, test_combined, y_test = load_aircraft_data(base_path)

# Prepare training features and target
X_train = train_combined.drop(columns=['engine_id', 'cycle', 'RUL'])
y_train = train_combined['RUL']

# Prepare test features
X_test = test_combined.drop(columns=['engine_id', 'cycle'])
