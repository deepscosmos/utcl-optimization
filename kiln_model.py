# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import TimeSeriesSplit
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# from keras.layers import Input, LSTM, Dense, LeakyReLU, Dropout, BatchNormalization
# from keras.models import Model
# from keras.optimizers import RMSprop
# from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# import optuna
# from genric import *
# from features import *

# # Model configuration
# model_name = 'Kiln_Torque_Model'
# target = get_model_target(model_name)[0]
# state_variables_raw = get_state_variables(model_name)
# positive_controls_var_raw = get_positive_controls(model_name)
# negative_controls_var_raw = get_negative_controls(model_name)
# all_variables_raw = get_all_variables(model_name)

# # Load and preprocess data
# raw_data = get_raw_data(all_variables_raw)
# for state_var in state_variables_raw:
#     calculate_shifted_EMA(raw_data, state_var, BP=5, span=15)
# for pos_control in positive_controls_var_raw:
#     calculate_TS_Feature(raw_data, pos_control, BP=5, span=15)
# for neg_control in negative_controls_var_raw:
#     calculate_TS_Feature(raw_data, neg_control, BP=5, span=15)
# calculate_EMA(raw_data, target, span=15)

# # Define input features for the neural network
# state_vars = [var + '_EMA_Shifted' for var in state_variables_raw]
# control_vars = [var + '_AUC' for var in positive_controls_var_raw + negative_controls_var_raw]
# target_var = target + '_EMA'

# # Create final dataset
# X_cols = state_vars + control_vars
# y_cols = target_var
# final_data = raw_data[X_cols + [y_cols]].dropna()

# # Split into training and test data
# train_rows = int(len(final_data) * 0.8)
# X_train = final_data[X_cols].iloc[:train_rows]
# y_train = final_data[y_cols].iloc[:train_rows]
# X_test = final_data[X_cols].iloc[train_rows:]
# y_test = final_data[y_cols].iloc[train_rows:]

# # Reshape data for LSTM (samples, time steps, features)
# def reshape_data_for_lstm(X_data):
#     return np.reshape(X_data.values, (X_data.shape[0], 1, X_data.shape[1]))

# X_train_lstm = reshape_data_for_lstm(X_train)
# X_test_lstm = reshape_data_for_lstm(X_test)

# # Define Optuna objective function to optimize hyperparameters
# def objective(trial):
#     # Hyperparameters to tune
#     lstm_units = trial.suggest_int('lstm_units', 32, 128, step=32)
#     dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5, step=0.05)
#     learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    
#     # Define LSTM-based Model
#     state_input = Input(shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), name='State_Input')

#     # LSTM layers
#     x = LSTM(lstm_units, activation='tanh', return_sequences=False)(state_input)
#     x = Dropout(dropout_rate)(x)
#     x = Dense(64)(x)
#     x = LeakyReLU(alpha=0.1)(x)
#     x = BatchNormalization()(x)
#     x = Dropout(dropout_rate)(x)
    
#     # Output layer
#     final_output = Dense(1, activation='linear')(x)

#     # Compile model
#     model = Model(inputs=state_input, outputs=final_output)
#     model.compile(optimizer=RMSprop(learning_rate=learning_rate), loss='mse', metrics=['mae'])

#     # Train model
#     early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#     checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)
#     reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=8, min_lr=0.0001)

#     history = model.fit(
#         X_train_lstm, y_train.values,
#         validation_data=(X_test_lstm, y_test.values),
#         epochs=100, batch_size=1000, verbose=0,
#         callbacks=[early_stopping, checkpoint, reduce_lr]
#     )
    
#     # Return validation loss for Optuna to minimize
#     return history.history['val_loss'][-1]

# # Create Optuna study and optimize hyperparameters
# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=50)

# # Best hyperparameters
# best_params = study.best_params
# print(f"Best hyperparameters: {best_params}")

# # Retrain model with best hyperparameters
# lstm_units = best_params['lstm_units']
# dropout_rate = best_params['dropout_rate']
# learning_rate = best_params['learning_rate']

# state_input = Input(shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), name='State_Input')

# # LSTM layers
# x = LSTM(lstm_units, activation='tanh', return_sequences=False)(state_input)
# x = Dropout(dropout_rate)(x)
# x = Dense(64)(x)
# x = LeakyReLU(alpha=0.1)(x)
# x = BatchNormalization()(x)
# x = Dropout(dropout_rate)(x)

# # Output layer
# final_output = Dense(1, activation='linear')(x)

# # Compile model
# final_model = Model(inputs=state_input, outputs=final_output)
# final_model.compile(optimizer=RMSprop(learning_rate=learning_rate), loss='mse', metrics=['mae'])

# # Train the final model
# history = final_model.fit(
#     X_train_lstm, y_train.values,
#     validation_data=(X_test_lstm, y_test.values),
#     epochs=100, batch_size=10000, verbose=2,
#     callbacks=[early_stopping, checkpoint, reduce_lr]
# )

# # Make predictions on the test data
# y_pred = final_model.predict(X_test_lstm)

# # Parity Plot (Predicted vs True values)
# plt.figure(figsize=(8, 8))
# plt.scatter(y_test, y_pred, alpha=0.6)
# plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # Ideal line
# plt.title('Parity Plot (True vs Predicted)')
# plt.xlabel('True Values')
# plt.ylabel('Predicted Values')
# plt.show()

# # Calculate and print the final MSE and MAE on the test set
# mse_test = mean_squared_error(y_test, y_pred)
# mae_test = mean_absolute_error(y_test, y_pred)

# print(f"Test MSE: {mse_test}")
# print(f"Test MAE: {mae_test}")



# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split, TimeSeriesSplit
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# import xgboost as xgb
# import optuna
# from feature import *
# from genric import *

# # Model configuration
# model_name = 'Kiln_Torque_Model'
# target = get_model_names(model_name)[0]
# state_variables_raw = get_state_variables(model_name)
# positive_controls_var_raw = get_positive_controls(model_name)
# negative_controls_var_raw = get_negative_controls(model_name)
# all_variables_raw = get_all_variables(model_name)

# # Load and preprocess data
# raw_data = get_raw_data(all_variables_raw)
# for state_var in state_variables_raw:
#     calculate_shifted_EMA(raw_data, state_var, BP=5, span=15)
# for pos_control in positive_controls_var_raw:
#     calculate_TS_Feature(raw_data, pos_control, BP=5, span=15)
# for neg_control in negative_controls_var_raw:
#     calculate_TS_Feature(raw_data, neg_control, BP=5, span=15)
# calculate_EMA(raw_data, target, span=15)

# def remove_outliers(data, columns):
#     for col in columns:
#         Q1 = data[col].quantile(0.25)
#         Q3 = data[col].quantile(0.75)
#         IQR = Q3 - Q1
#         lower_bound = Q1 - 1.5 * IQR
#         upper_bound = Q3 + 1.5 * IQR
#         data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
#     return data

# # Define input features for the model
# state_vars = [var + '_EMA_Shifted' for var in state_variables_raw]
# control_vars = [var + '_AUC' for var in positive_controls_var_raw + negative_controls_var_raw]
# target_var = target + '_EMA'

# # Create final dataset
# X_cols = state_vars + control_vars
# y_cols = target_var
# final_data = raw_data[X_cols + [y_cols]].dropna()

# # Split into training and test data
# train_rows = int(len(final_data) * 0.8)
# X_train = final_data[X_cols].iloc[:train_rows].values
# y_train = final_data[y_cols].iloc[:train_rows].values
# X_test = final_data[X_cols].iloc[train_rows:].values
# y_test = final_data[y_cols].iloc[train_rows:].values

# # Define Optuna objective function
# def objective(trial):
#     params = {
#         'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
#         'max_depth': trial.suggest_int('max_depth', 3, 10),
#         'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
#         'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
#         'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
#         'gamma': trial.suggest_loguniform('gamma', 1e-6, 1.0),
#         'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-6, 1.0),
#         'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-6, 1.0)
#     }

#     model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, **params)
    
#     # Cross-validation with time series split
#     tscv = TimeSeriesSplit(n_splits=3)
#     scores = []
#     for train_index, val_index in tscv.split(X_train):
#         X_tr, X_val = X_train[train_index], X_train[val_index]
#         y_tr, y_val = y_train[train_index], y_train[val_index]
#         model.fit(X_tr, y_tr)
#         y_pred_val = model.predict(X_val)
#         mse = mean_squared_error(y_val, y_pred_val)
#         scores.append(mse)
        
#     return np.mean(scores)

# # Run Optuna optimization
# study = optuna.create_study(direction="minimize")
# study.optimize(objective, n_trials=50)

# # Train the best model
# best_params = study.best_params
# print("Best Parameters: ", best_params)

# best_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, **best_params)
# best_model.fit(X_train, y_train)

# # Predict on test set
# y_pred = best_model.predict(X_test)

# # Parity Plot (Predicted vs True values)
# plt.figure(figsize=(8, 8))
# plt.scatter(y_test, y_pred, alpha=0.6)
# plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
# plt.title('Parity Plot (True vs Predicted)')
# plt.xlabel('True Values')
# plt.ylabel('Predicted Values')
# plt.show()

# # Calculate and print the final MSE and MAE on the test set
# mse_test = mean_squared_error(y_test, y_pred)
# mae_test = mean_absolute_error(y_test, y_pred)

# print(f"Test MSE: {mse_test}")
# print(f"Test MAE: {mae_test}")

# # Feature importance plot
# xgb.plot_importance(best_model)
# plt.title('Feature Importance')
# plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from keras.layers import Input, LSTM, Dense, LeakyReLU, Dropout, BatchNormalization
from keras.models import Model, Sequential
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import xgboost as xgb
import optuna
from CalculateFeature import *
from get_data import *

# **Model Configuration**
model_name = 'Kiln_Torque_Model'
target = get_model_target(model_name)[0]
state_variables_raw = get_state_variables(model_name)
positive_controls_var_raw = get_positive_controls(model_name)
negative_controls_var_raw = get_negative_controls(model_name)
all_variables_raw = get_all_variables(model_name)

# **Load and preprocess data**
raw_data = get_raw_data(all_variables_raw)
for state_var in state_variables_raw:
    calculate_shifted_EMA(raw_data, state_var, BP=5, span=15)
for pos_control in positive_controls_var_raw:
    calculate_TS_Feature(raw_data, pos_control, BP=5, span=15)
for neg_control in negative_controls_var_raw:
    calculate_TS_Feature(raw_data, neg_control, BP=5, span=15)
calculate_EMA(raw_data, target, span=15)

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

# **Reshape data for LSTM**
def reshape_data_for_lstm(X_data):
    return np.reshape(X_data, (X_data.shape[0], 1, X_data.shape[1]))

X_train_lstm = reshape_data_for_lstm(X_train)
X_test_lstm = reshape_data_for_lstm(X_test)

# **LSTM Model**
lstm_model = Sequential([
    LSTM(64, activation='tanh', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
    Dropout(0.3),
    Dense(64),
    LeakyReLU(alpha=0.1),
    Dense(1, activation='linear')
])
lstm_model.compile(optimizer=RMSprop(learning_rate=0.001), loss='mse', metrics=['mae'])
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train LSTM
lstm_model.fit(X_train_lstm, y_train, validation_data=(X_test_lstm, y_test), epochs=50, batch_size=64, callbacks=[early_stopping], verbose=1)
y_pred_lstm = lstm_model.predict(X_test_lstm)
mae_lstm = mean_absolute_error(y_test, y_pred_lstm)

# **ANN Model**
ann_model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1, activation='linear')
])
ann_model.compile(optimizer=RMSprop(learning_rate=0.001), loss='mse', metrics=['mae'])

# Train ANN
ann_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=64, callbacks=[early_stopping], verbose=1)
y_pred_ann = ann_model.predict(X_test)
mae_ann = mean_absolute_error(y_test, y_pred_ann)

# **Random Forest Model**
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train.ravel())
y_pred_rf = rf_model.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)

# **XGBoost Model**
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

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)
best_params = study.best_params
print("Best Parameters: ", best_params)

xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, **best_params)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)

# **MAE Results**
print(f"MAE for LSTM: {mae_lstm}")
print(f"MAE for ANN: {mae_ann}")
print(f"MAE for Random Forest: {mae_rf}")
print(f"MAE for XGBoost: {mae_xgb}")

# **Parity Plots**
models = ['LSTM', 'ANN', 'Random Forest', 'XGBoost']
predictions = [y_pred_lstm, y_pred_ann, y_pred_rf, y_pred_xgb]

for i, (model_name, y_pred) in enumerate(zip(models, predictions)):
    plt.figure(i)
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.title(f'Parity Plot: {model_name}')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.show()
