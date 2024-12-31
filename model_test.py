import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import optuna
from feature import *
from genric import *

# Model configuration
model_name = 'Kiln_Torque_Model'
target = get_model_names(model_name)[0]
state_variables_raw = get_state_variables(model_name)
positive_controls_var_raw = get_positive_controls(model_name)
negative_controls_var_raw = get_negative_controls(model_name)
all_variables_raw = get_all_variables(model_name)

# Load and preprocess data
raw_data = get_raw_data(all_variables_raw)
for state_var in state_variables_raw:
    calculate_shifted_EMA(raw_data, state_var, BP=5, span=15)
for pos_control in positive_controls_var_raw:
    calculate_TS_Feature(raw_data, pos_control, BP=5, span=15)
for neg_control in negative_controls_var_raw:
    calculate_TS_Feature(raw_data, neg_control, BP=5, span=15)
calculate_EMA(raw_data, target, span=15)

def remove_outliers(data, columns):
    for col in columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
    return data

# Define input features for the model
state_vars = [var + '_EMA_Shifted' for var in state_variables_raw]
control_vars = [var + '_AUC' for var in positive_controls_var_raw + negative_controls_var_raw]
target_var = target + '_EMA'

# Create final dataset
X_cols = state_vars + control_vars
y_cols = target_var
final_data = raw_data[X_cols + [y_cols]].dropna()

# Split into training and test data
train_rows = int(len(final_data) * 0.8)
X_train = final_data[X_cols].iloc[:train_rows].values
y_train = final_data[y_cols].iloc[:train_rows].values
X_test = final_data[X_cols].iloc[train_rows:].values
y_test = final_data[y_cols].iloc[train_rows:].values

# Define Optuna objective function
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_loguniform('gamma', 1e-6, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-6, 1.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-6, 1.0)
    }

    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, **params)
    
    # Cross-validation with time series split
    tscv = TimeSeriesSplit(n_splits=3)
    scores = []
    for train_index, val_index in tscv.split(X_train):
        X_tr, X_val = X_train[train_index], X_train[val_index]
        y_tr, y_val = y_train[train_index], y_train[val_index]
        model.fit(X_tr, y_tr)
        y_pred_val = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred_val)
        scores.append(mse)
        
    return np.mean(scores)

# Run Optuna optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

# Train the best model
best_params = study.best_params
print("Best Parameters: ", best_params)

best_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, **best_params)
best_model.fit(X_train, y_train)

# Predict on test set
y_pred = best_model.predict(X_test)

# Parity Plot (Predicted vs True values)
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Parity Plot (True vs Predicted)')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.show()

# Calculate and print the final MSE and MAE on the test set
mse_test = mean_squared_error(y_test, y_pred)
mae_test = mean_absolute_error(y_test, y_pred)

print(f"Test MSE: {mse_test}")
print(f"Test MAE: {mae_test}")

# Feature importance plot
xgb.plot_importance(best_model)
plt.title('Feature Importance')
plt.show()
