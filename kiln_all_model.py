# import pandas as pd 
# import numpy as np 
# import matplotlib.pyplot as plt 
# from sklearn.model_selection import train_test_split, TimeSeriesSplit
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# from sklearn.ensemble import RandomForestRegressor
# from keras.layers import Input, LSTM, Dense, LeakyReLU, Dropout, BatchNormalization
# from keras.models import Model, Sequential
# from keras.optimizers import RMSprop
# from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# import xgboost as xgb
# import optuna
# from CalculateFeature import *
# from get_data import *

# # **Model Configuration**
# model_name = 'Kiln_Torque_Model'
# target = get_model_target(model_name)[0]
# state_variables_raw = get_state_variables(model_name)
# positive_controls_var_raw = get_positive_controls(model_name)
# negative_controls_var_raw = get_negative_controls(model_name)
# all_variables_raw = get_all_variables(model_name)

# # **Load and preprocess data**
# raw_data = get_raw_data(all_variables_raw)
# for state_var in state_variables_raw:
#     calculate_shifted_EMA(raw_data, state_var, BP=5, span=15)
# for pos_control in positive_controls_var_raw:
#     calculate_TS_Feature(raw_data, pos_control, BP=5, span=15)
# for neg_control in negative_controls_var_raw:
#     calculate_TS_Feature(raw_data, neg_control, BP=5, span=15)
# calculate_EMA(raw_data, target, span=15)

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

# # **Reshape data for LSTM**
# def reshape_data_for_lstm(X_data):
#     return np.reshape(X_data, (X_data.shape[0], 1, X_data.shape[1]))

# X_train_lstm = reshape_data_for_lstm(X_train)
# X_test_lstm = reshape_data_for_lstm(X_test)

# # **LSTM Model**
# lstm_model = Sequential([
#     LSTM(64, activation='tanh', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
#     Dropout(0.3),
#     Dense(64),
#     LeakyReLU(alpha=0.1),
#     Dense(1, activation='linear')
# ])
# lstm_model.compile(optimizer=RMSprop(learning_rate=0.001), loss='mse', metrics=['mae'])
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# # Train LSTM
# lstm_model.fit(X_train_lstm, y_train, validation_data=(X_test_lstm, y_test), epochs=50, batch_size=64, callbacks=[early_stopping], verbose=1)
# y_pred_lstm = lstm_model.predict(X_test_lstm)
# mae_lstm = mean_absolute_error(y_test, y_pred_lstm)
# mse_lstm = mean_squared_error(y_test, y_pred_lstm)

# # **ANN Model**
# ann_model = Sequential([
#     Dense(128, input_dim=X_train.shape[1], activation='relu'),
#     Dropout(0.3),
#     Dense(64, activation='relu'),
#     Dense(1, activation='linear')
# ])
# ann_model.compile(optimizer=RMSprop(learning_rate=0.001), loss='mse', metrics=['mae'])

# # Train ANN
# ann_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=64, callbacks=[early_stopping], verbose=1)
# y_pred_ann = ann_model.predict(X_test)
# mae_ann = mean_absolute_error(y_test, y_pred_ann)
# mse_ann = mean_squared_error(y_test, y_pred_ann)

# # **Random Forest Model**
# rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
# rf_model.fit(X_train, y_train.ravel())
# y_pred_rf = rf_model.predict(X_test)
# mae_rf = mean_absolute_error(y_test, y_pred_rf)
# mse_rf = mean_squared_error(y_test, y_pred_rf)

# # **XGBoost Model**
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

# study = optuna.create_study(direction="minimize")
# study.optimize(objective, n_trials=50)
# best_params = study.best_params
# print("Best Parameters: ", best_params)

# xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, **best_params)
# xgb_model.fit(X_train, y_train)
# y_pred_xgb = xgb_model.predict(X_test)
# mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
# mse_xgb = mean_squared_error(y_test, y_pred_xgb)

# # **MAE and MSE Results**
# print(f"MAE for LSTM: {mae_lstm}")
# print(f"MAE for ANN: {mae_ann}")
# print(f"MAE for Random Forest: {mae_rf}")
# print(f"MAE for XGBoost: {mae_xgb}")

# print(f"MSE for LSTM: {mse_lstm}")
# print(f"MSE for ANN: {mse_ann}")
# print(f"MSE for Random Forest: {mse_rf}")
# print(f"MSE for XGBoost: {mse_xgb}")

# # **Bar Plot for MAE and MSE**
# models = ['LSTM', 'ANN', 'Random Forest', 'XGBoost']
# mae_values = [mae_lstm, mae_ann, mae_rf, mae_xgb]
# mse_values = [mse_lstm, mse_ann, mse_rf, mse_xgb]

# # Set up the bar plot
# x = np.arange(len(models))  # model labels
# width = 0.35  # width of the bars

# # Create subplots
# fig, ax = plt.subplots(figsize=(10, 6))

# # Plot MAE and MSE
# bars1 = ax.bar(x - width/2, mae_values, width, label='MAE', color='lightblue')
# bars2 = ax.bar(x + width/2, mse_values, width, label='MSE', color='lightcoral')

# # Add some text for labels, title, and custom x-axis tick labels
# ax.set_xlabel('Models')
# ax.set_ylabel('Error')
# ax.set_title('Comparison of MAE and MSE for Different Models')
# ax.set_xticks(x)
# ax.set_xticklabels(models)
# ax.legend()

# # Display the values on top of the bars
# def add_value_labels(bars):
#     for bar in bars:
#         height = bar.get_height()
#         ax.annotate(f'{height:.2f}',
#                     xy=(bar.get_x() + bar.get_width() / 2, height),
#                     xytext=(0, 3),  # 3 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom')

# # Call the function to add labels
# add_value_labels(bars1)
# add_value_labels(bars2)

# # Show plot
# plt.tight_layout()
# plt.show()

# # **Parity Plots**
# predictions = [y_pred_lstm, y_pred_ann, y_pred_rf, y_pred_xgb]

# for i, (model_name, y_pred) in enumerate(zip(models, predictions)):
#     plt.figure(i)
#     plt.scatter(y_test, y_pred)
#     plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
#     plt.xlabel('True Values')
#     plt.ylabel('Predictions')
#     plt.title(f'Parity Plot for {model_name}')
#     plt.show()




# import pandas as pd 
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split, TimeSeriesSplit
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# from sklearn.ensemble import RandomForestRegressor
# from keras.layers import Input, LSTM, Dense, LeakyReLU, Dropout, BatchNormalization
# from keras.models import Model, Sequential
# from keras.optimizers import RMSprop
# from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# import xgboost as xgb
# import optuna
# from CalculateFeature import *
# from get_data import *

# # **Model Configuration**
# model_name = 'Kiln_Torque_Model'
# target = get_model_target(model_name)[0]
# state_variables_raw = get_state_variables(model_name)
# positive_controls_var_raw = get_positive_controls(model_name)
# negative_controls_var_raw = get_negative_controls(model_name)
# all_variables_raw = get_all_variables(model_name)

# # **Load and preprocess data**
# raw_data = get_raw_data(all_variables_raw)
# for state_var in state_variables_raw:
#     calculate_shifted_EMA(raw_data, state_var, BP=5, span=15)
# for pos_control in positive_controls_var_raw:
#     calculate_TS_Feature(raw_data, pos_control, BP=5, span=15)
# for neg_control in negative_controls_var_raw:
#     calculate_TS_Feature(raw_data, neg_control, BP=5, span=15)
# calculate_EMA(raw_data, target, span=15)

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

# # **Reshape data for LSTM**
# def reshape_data_for_lstm(X_data):
#     return np.reshape(X_data, (X_data.shape[0], 1, X_data.shape[1]))

# X_train_lstm = reshape_data_for_lstm(X_train)
# X_test_lstm = reshape_data_for_lstm(X_test)

# # **LSTM Model**
# lstm_model = Sequential([
#     LSTM(64, activation='tanh', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
#     Dropout(0.3),
#     Dense(64),
#     LeakyReLU(alpha=0.1),
#     Dense(1, activation='linear')
# ])
# lstm_model.compile(optimizer=RMSprop(learning_rate=0.001), loss='mse', metrics=['mae'])
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# # Train LSTM
# lstm_model.fit(X_train_lstm, y_train, validation_data=(X_test_lstm, y_test), epochs=50, batch_size=64, callbacks=[early_stopping], verbose=1)
# y_pred_lstm = lstm_model.predict(X_test_lstm)
# mae_lstm = mean_absolute_error(y_test, y_pred_lstm)
# mse_lstm = mean_squared_error(y_test, y_pred_lstm)

# # **ANN Model Optimization with Optuna**
# def create_ann_model(trial):
#     model = Sequential()
    
#     # Number of layers
#     n_layers = trial.suggest_int('n_layers', 1, 3)
    
#     # Adding layers with neurons
#     for i in range(n_layers):
#         n_units = trial.suggest_int(f'n_units_l{i}', 64, 256)  # Number of neurons in each layer
#         model.add(Dense(n_units, activation='relu'))
#         model.add(Dropout(trial.suggest_float('dropout', 0.2, 0.5)))  # Dropout rate for regularization

#     model.add(Dense(1, activation='linear'))  # Output layer
#     model.compile(optimizer=RMSprop(learning_rate=trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)),
#                   loss='mse', metrics=['mae'])
#     return model

# # Objective function for Optuna
# def objective_ann(trial):
#     model = create_ann_model(trial)
    
#     # Use EarlyStopping to avoid overfitting
#     early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
#     # Train the model
#     model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=64, callbacks=[early_stopping], verbose=0)
    
#     # Evaluate the model on the validation set
#     y_pred = model.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     return mse

# # Optuna optimization for ANN
# study_ann = optuna.create_study(direction="minimize")
# study_ann.optimize(objective_ann, n_trials=50)
# best_params_ann = study_ann.best_params
# print("Best Parameters for ANN: ", best_params_ann)

# # Train the best ANN model
# best_ann_model = create_ann_model(best_params_ann)
# best_ann_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=64, callbacks=[early_stopping], verbose=1)
# y_pred_ann = best_ann_model.predict(X_test)
# mae_ann = mean_absolute_error(y_test, y_pred_ann)
# mse_ann = mean_squared_error(y_test, y_pred_ann)

# # **Random Forest Model**
# rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
# rf_model.fit(X_train, y_train.ravel())
# y_pred_rf = rf_model.predict(X_test)
# mae_rf = mean_absolute_error(y_test, y_pred_rf)
# mse_rf = mean_squared_error(y_test, y_pred_rf)

# # **XGBoost Model with Optuna Optimization**
# def objective_xgb(trial):
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

# study_xgb = optuna.create_study(direction="minimize")
# study_xgb.optimize(objective_xgb, n_trials=50)
# best_params_xgb = study_xgb.best_params
# print("Best Parameters for XGBoost: ", best_params_xgb)

# xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, **best_params_xgb)
# xgb_model.fit(X_train, y_train)
# y_pred_xgb = xgb_model.predict(X_test)
# mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
# mse_xgb = mean_squared_error(y_test, y_pred_xgb)

# # **MAE and MSE Results**
# print(f"MAE for LSTM: {mae_lstm}")
# print(f"MAE for ANN: {mae_ann}")
# print(f"MAE for Random Forest: {mae_rf}")
# print(f"MAE for XGBoost: {mae_xgb}")

# print(f"MSE for LSTM: {mse_lstm}")
# print(f"MSE for ANN: {mse_ann}")
# print(f"MSE for Random Forest: {mse_rf}")
# print(f"MSE for XGBoost: {mse_xgb}")

# # **Visualizations of Predictions**
# plt.figure(figsize=(10,6))
# plt.plot(y_test, label='True')
# plt.plot(y_pred_lstm, label='LSTM Prediction')
# plt.plot(y_pred_ann, label='ANN Prediction')
# plt.plot(y_pred_rf, label='RF Prediction')
# plt.plot(y_pred_xgb, label='XGB Prediction')
# plt.legend()
# plt.title("True vs Predicted Values")
# plt.show()


# import pandas as pd  
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split, TimeSeriesSplit
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import StandardScaler
# from keras.layers import Input, LSTM, Dense, LeakyReLU, Dropout
# from keras.models import Model, Sequential
# from keras.optimizers import RMSprop
# from keras.callbacks import EarlyStopping
# import xgboost as xgb
# from CalculateFeature import *
# from get_data import *

# # **Model Configuration**
# model_name = 'Clinker_Temp_Model'
# target = get_model_target(model_name)[0]
# state_variables_raw = get_state_variables(model_name)
# positive_controls_var_raw = get_positive_controls(model_name)
# negative_controls_var_raw = get_negative_controls(model_name)
# all_variables_raw = get_all_variables(model_name)

# # **Load and preprocess data**
# raw_data = get_raw_data(all_variables_raw)
# for state_var in state_variables_raw:
#     calculate_shifted_EMA(raw_data, state_var, BP=5, span=15)
# for pos_control in positive_controls_var_raw:
#     calculate_TS_Feature(raw_data, pos_control, BP=5, span=15)
# for neg_control in negative_controls_var_raw:
#     calculate_TS_Feature(raw_data, neg_control, BP=5, span=15)
# calculate_EMA(raw_data, target, span=15)

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

# # **Reshape data for LSTM**
# def reshape_data_for_lstm(X_data):
#     return np.reshape(X_data, (X_data.shape[0], 1, X_data.shape[1]))

# X_train_lstm = reshape_data_for_lstm(X_train)
# X_test_lstm = reshape_data_for_lstm(X_test)

# # **LSTM Model**
# lstm_model = Sequential([
#     LSTM(64, activation='tanh', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
#     Dropout(0.3),
#     Dense(64),
#     LeakyReLU(alpha=0.1),
#     Dense(1, activation='linear')
# ])
# lstm_model.compile(optimizer=RMSprop(learning_rate=0.001), loss='mse', metrics=['mae'])
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# # Train LSTM
# lstm_model.fit(X_train_lstm, y_train, validation_data=(X_test_lstm, y_test), epochs=50, batch_size=64, callbacks=[early_stopping], verbose=1)
# y_pred_lstm = lstm_model.predict(X_test_lstm)
# mae_lstm = mean_absolute_error(y_test, y_pred_lstm)
# mse_lstm = mean_squared_error(y_test, y_pred_lstm)

# # **Scale data for ANN**
# scaler_X = StandardScaler()
# scaler_y = StandardScaler()

# X_train_scaled = scaler_X.fit_transform(X_train)
# X_test_scaled = scaler_X.transform(X_test)

# y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
# y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

# # **ANN Model**
# ann_model = Sequential([
#     Dense(128, activation='relu', input_dim=X_train_scaled.shape[1]),
#     Dropout(0.3),
#     Dense(64, activation='relu'),
#     Dropout(0.3),
#     Dense(1, activation='linear')
# ])

# ann_model.compile(optimizer=RMSprop(learning_rate=0.001), loss='mse', metrics=['mae'])
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# # Train ANN
# ann_model.fit(X_train_scaled, y_train_scaled, validation_data=(X_test_scaled, y_test_scaled), epochs=50, batch_size=64, callbacks=[early_stopping], verbose=1)
# y_pred_ann_scaled = ann_model.predict(X_test_scaled)

# # Inverse transform predictions and true values
# y_pred_ann = scaler_y.inverse_transform(y_pred_ann_scaled)
# y_test_original = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1))

# # Calculate metrics on original scale
# mae_ann = mean_absolute_error(y_test_original, y_pred_ann)
# mse_ann = mean_squared_error(y_test_original, y_pred_ann)

# # **Random Forest Model**
# rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
# rf_model.fit(X_train, y_train.ravel())
# y_pred_rf = rf_model.predict(X_test)
# mae_rf = mean_absolute_error(y_test, y_pred_rf)
# mse_rf = mean_squared_error(y_test, y_pred_rf)

# # **XGBoost Model**
# xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42)
# xgb_model.fit(X_train, y_train)
# y_pred_xgb = xgb_model.predict(X_test)
# mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
# mse_xgb = mean_squared_error(y_test, y_pred_xgb)

# # **MAE and MSE Results**
# print(f"MAE for LSTM: {mae_lstm}")
# print(f"MAE for ANN: {mae_ann}")
# print(f"MAE for Random Forest: {mae_rf}")
# print(f"MAE for XGBoost: {mae_xgb}")

# print(f"MSE for LSTM: {mse_lstm}")
# print(f"MSE for ANN: {mse_ann}")
# print(f"MSE for Random Forest: {mse_rf}")
# print(f"MSE for XGBoost: {mse_xgb}")

# # Plot MSE and MAE Comparison
# models = ['LSTM', 'ANN', 'Random Forest', 'XGBoost']
# mse_values = [mse_lstm, mse_ann, mse_rf, mse_xgb]
# mae_values = [mae_lstm, mae_ann, mae_rf, mae_xgb]

# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.bar(models, mse_values, color='skyblue')
# plt.title('MSE Comparison')
# plt.ylabel('Mean Squared Error')

# plt.subplot(1, 2, 2)
# plt.bar(models, mae_values, color='salmon')
# plt.title('MAE Comparison')
# plt.ylabel('Mean Absolute Error')
# plt.tight_layout()
# plt.show()

# # Parity Plots for Each Model
# def parity_plot(y_true, y_pred, model_name):
#     plt.figure(figsize=(6, 6))
#     plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='k')
#     plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--', linewidth=2)
#     plt.title(f'Parity Plot for {model_name}')
#     plt.xlabel('Actual Values')
#     plt.ylabel('Predicted Values')
#     plt.grid(True)
#     plt.show()

# # LSTM Parity Plot
# parity_plot(y_test, y_pred_lstm, 'LSTM')

# # ANN Parity Plot
# parity_plot(y_test_original, y_pred_ann, 'ANN')

# # Random Forest Parity Plot
# parity_plot(y_test, y_pred_rf, 'Random Forest')

# # XGBoost Parity Plot
# parity_plot(y_test, y_pred_xgb, 'XGBoost')



# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.dummy import DummyRegressor
# from keras.layers import Input, LSTM, Dense, LeakyReLU, Dropout
# from keras.models import Model, Sequential
# from keras.optimizers import RMSprop
# from keras.callbacks import EarlyStopping
# import xgboost as xgb
# from CalculateFeature import *
# from get_data import *

# # **Model Configuration**
# model_name = 'BZT_Model'
# target = get_model_target(model_name)[0]
# state_variables_raw = get_state_variables(model_name)
# positive_controls_var_raw = get_positive_controls(model_name)
# negative_controls_var_raw = get_negative_controls(model_name)
# all_variables_raw = get_all_variables(model_name)

# # **Load and preprocess data**
# raw_data = get_raw_data(all_variables_raw)
# for state_var in state_variables_raw:
#     calculate_shifted_EMA(raw_data, state_var, BP=5, span=15)
# for pos_control in positive_controls_var_raw:
#     calculate_TS_Feature(raw_data, pos_control, BP=5, span=15)
# for neg_control in negative_controls_var_raw:
#     calculate_TS_Feature(raw_data, neg_control, BP=5, span=15)
# calculate_EMA(raw_data, target, span=15)

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

# # **Dummy Model for Baseline Comparison**
# dummy_model = DummyRegressor(strategy='mean')
# dummy_model.fit(X_train, y_train)
# y_pred_dummy = dummy_model.predict(X_test)
# mae_dummy = mean_absolute_error(y_test, y_pred_dummy)

# # **Reshape data for LSTM**
# def reshape_data_for_lstm(X_data):
#     return np.reshape(X_data, (X_data.shape[0], 1, X_data.shape[1]))

# X_train_lstm = reshape_data_for_lstm(X_train)
# X_test_lstm = reshape_data_for_lstm(X_test)

# # **LSTM Model**
# lstm_model = Sequential([
#     LSTM(64, activation='tanh', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
#     Dropout(0.3),
#     Dense(64),
#     LeakyReLU(alpha=0.1),
#     Dense(1, activation='linear')
# ])
# lstm_model.compile(optimizer=RMSprop(learning_rate=0.001), loss='mse', metrics=['mae'])
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# # Train LSTM
# lstm_model.fit(X_train_lstm, y_train, validation_data=(X_test_lstm, y_test), epochs=50, batch_size=64, callbacks=[early_stopping], verbose=1)
# y_pred_lstm = lstm_model.predict(X_test_lstm)
# mae_lstm = mean_absolute_error(y_test, y_pred_lstm)

# # **Scale data for ANN**
# scaler_X = StandardScaler()
# scaler_y = StandardScaler()

# X_train_scaled = scaler_X.fit_transform(X_train)
# X_test_scaled = scaler_X.transform(X_test)

# y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
# y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

# # **ANN Model**
# ann_model = Sequential([
#     Dense(128, activation='relu', input_dim=X_train_scaled.shape[1]),
#     Dropout(0.3),
#     Dense(64, activation='relu'),
#     Dropout(0.3),
#     Dense(1, activation='linear')
# ])

# ann_model.compile(optimizer=RMSprop(learning_rate=0.001), loss='mse', metrics=['mae'])
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# # Train ANN
# ann_model.fit(X_train_scaled, y_train_scaled, validation_data=(X_test_scaled, y_test_scaled), epochs=50, batch_size=64, callbacks=[early_stopping], verbose=1)
# y_pred_ann_scaled = ann_model.predict(X_test_scaled)

# # Inverse transform predictions and true values
# y_pred_ann = scaler_y.inverse_transform(y_pred_ann_scaled)
# y_test_original = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1))

# # Calculate MAE on original scale
# mae_ann = mean_absolute_error(y_test_original, y_pred_ann)

# # **Random Forest Model**
# rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
# rf_model.fit(X_train, y_train.ravel())
# y_pred_rf = rf_model.predict(X_test)
# mae_rf = mean_absolute_error(y_test, y_pred_rf)

# # **XGBoost Model**
# xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42)
# xgb_model.fit(X_train, y_train)
# y_pred_xgb = xgb_model.predict(X_test)
# mae_xgb = mean_absolute_error(y_test, y_pred_xgb)

# # **MAE Results**
# print(f"MAE for Dummy Model (Baseline): {mae_dummy}")
# print(f"MAE for LSTM: {mae_lstm}")
# print(f"MAE for ANN: {mae_ann}")
# print(f"MAE for Random Forest: {mae_rf}")
# print(f"MAE for XGBoost: {mae_xgb}")

# # Plot MAE Comparison
# models = ['Dummy', 'LSTM', 'ANN', 'Random Forest', 'XGBoost']
# mae_values = [mae_dummy, mae_lstm, mae_ann, mae_rf, mae_xgb]

# plt.figure(figsize=(8, 6))
# plt.bar(models, mae_values, color='skyblue')
# plt.title('MAE Comparison')
# plt.ylabel('Mean Absolute Error')
# plt.tight_layout()
# plt.show()

# # Parity Plots for Each Model
# def parity_plot(y_true, y_pred, model_name):
#     plt.figure(figsize=(6, 6))
#     plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='k')
#     plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--', linewidth=2)
#     plt.title(f'Parity Plot for {model_name}')
#     plt.xlabel('Actual Values')
#     plt.ylabel('Predicted Values')
#     plt.grid(True)
#     plt.show()

# # Dummy Parity Plot
# parity_plot(y_test, y_pred_dummy, 'Dummy')

# # LSTM Parity Plot
# parity_plot(y_test, y_pred_lstm, 'LSTM')

# # ANN Parity Plot
# parity_plot(y_test_original, y_pred_ann, 'ANN')

# # Random Forest Parity Plot
# parity_plot(y_test, y_pred_rf, 'Random Forest')

# # XGBoost Parity Plot
# parity_plot(y_test, y_pred_xgb, 'XGBoost')


# # Print state variables, control variables, and features
# print("\n=== Model Variables and Features ===\n")
# print("Target Variable:", target)
# print("\nRaw State Variables:")
# print(state_variables_raw)

# print("\nPositive Control Variables:")
# print(positive_controls_var_raw)

# print("\nNegative Control Variables:")
# print(negative_controls_var_raw)

# print("\nAll Variables:")
# print(all_variables_raw)

# print("\n=== Feature Calculation Details ===\n")
# print("State Variable Features (Exponential Moving Average + Shift):")
# for var in state_vars:
#     print(f"Feature: {var} (calculated from state variable with EMA and shift)")

# print("\nControl Variable Features (Area Under Curve):")
# for var in control_vars:
#     print(f"Feature: {var} (calculated from control variable with AUC)")

# print("\nTarget Variable Feature (Exponential Moving Average):")
# print(f"Feature: {target_var} (calculated from target variable with EMA)")

# print("\nFinal Input Features for the Model:")
# print(X_cols)

# print("\nFinal Target Variable for the Model:")
# print(y_cols)

# print("\n=== Dataset Overview ===\n")
# print("First 5 Rows of Final Dataset:")
# print(final_data.head())

# print("\nNumber of Features Used for Training:", len(X_cols))
# print("Number of Data Points in Final Dataset:", len(final_data))

# print("\n=== Data Splits ===\n")
# print(f"Number of Training Samples: {len(X_train)}")
# print(f"Number of Test Samples: {len(X_test)}")



# import pandas as pd 
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.dummy import DummyRegressor
# from keras.layers import Input, LSTM, Dense, LeakyReLU, Dropout
# from keras.models import Sequential
# from keras.optimizers import RMSprop
# from keras.callbacks import EarlyStopping
# import xgboost as xgb
# from CalculateFeature import *
# from get_data import *

# # **Model Configuration**
# model_name = 'BZT_Model'
# target = get_model_target(model_name)[0]
# state_variables_raw = get_state_variables(model_name)
# positive_controls_var_raw = get_positive_controls(model_name)
# negative_controls_var_raw = get_negative_controls(model_name)
# all_variables_raw = get_all_variables(model_name)

# # **Load and preprocess data**
# raw_data = get_raw_data(all_variables_raw)

# # Print raw data head
# print("\n=== Raw Data (First 5 Rows) ===")
# print(raw_data.head())

# # # **Datetime Handling: Convert 'Time' column to datetime format**
# # if 'Time' in raw_data.columns:
# #     try:
# #         # Convert 'Time' column to datetime with error handling
# #         raw_data['Time'] = pd.to_datetime(raw_data['Time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        
# #         # Check for invalid datetime values
# #         if raw_data['Time'].isnull().any():
# #             print("Warning: Some 'Time' values could not be converted. These rows will be dropped.")
# #             print(raw_data[raw_data['Time'].isnull()])
# #             # Drop rows with invalid 'Time' values
# #             raw_data = raw_data.dropna(subset=['Time'])
            
# #     except Exception as e:
# #         print(f"Error while converting 'Time' column: {e}")
# #         raise
# # else:
# #     print("Error: 'Time' column is missing in the dataset.")
# #     print("Available columns:", raw_data.columns)
# #     raise KeyError("'Time' column is required but not found in the dataset.")

# # # Set 'Time' as the index
# # raw_data.set_index('Time', inplace=True)

# # # Print to verify the datetime conversion
# # print("\n=== Raw Data After Datetime Conversion ===")
# # print(raw_data.head())
# # print(raw_data.shape)

# # Apply feature calculations for each state variable
# for state_var in state_variables_raw:
#     calculate_shifted_EMA(raw_data, state_var, BP=5, span=15)

# # Apply feature calculations for positive and negative controls
# for pos_control in positive_controls_var_raw:
#     calculate_TS_Feature(raw_data, pos_control, BP=5, span=15)
# for neg_control in negative_controls_var_raw:
#     calculate_TS_Feature(raw_data, neg_control, BP=5, span=15)

# # Apply EMA calculation for the target variable
# calculate_EMA(raw_data, target, span=15)

# # **Create final dataset with features and target variable**
# state_vars = [var + '_EMA_Shifted' for var in state_variables_raw]
# control_vars = [var + '_AUC' for var in positive_controls_var_raw + negative_controls_var_raw]
# target_var = target + '_EMA'

# # Create the final dataset by combining input features and target variable
# final_data = raw_data[state_vars + control_vars + [target_var]].dropna()

# # Print final data head
# print("\n=== Final Data (First 5 Rows) ===")
# print(final_data.head())

# # **Split data into training and testing sets**
# train_rows = int(len(final_data) * 0.8)
# X_train = final_data[state_vars + control_vars].iloc[:train_rows].values
# y_train = final_data[target_var].iloc[:train_rows].values
# X_test = final_data[state_vars + control_vars].iloc[train_rows:].values
# y_test = final_data[target_var].iloc[train_rows:].values

# # **Dummy Model for Baseline Comparison**
# dummy_model = DummyRegressor(strategy='mean')
# dummy_model.fit(X_train, y_train)
# y_pred_dummy = dummy_model.predict(X_test)
# mae_dummy = mean_absolute_error(y_test, y_pred_dummy)

# # **Reshape data for LSTM**
# def reshape_data_for_lstm(X_data):
#     return np.reshape(X_data, (X_data.shape[0], 1, X_data.shape[1]))

# X_train_lstm = reshape_data_for_lstm(X_train)
# X_test_lstm = reshape_data_for_lstm(X_test)

# # **LSTM Model**
# lstm_model = Sequential([
#     LSTM(64, activation='tanh', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
#     Dropout(0.3),
#     Dense(64),
#     LeakyReLU(alpha=0.1),
#     Dense(1, activation='linear')
# ])
# lstm_model.compile(optimizer=RMSprop(learning_rate=0.001), loss='mse', metrics=['mae'])
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# # Train LSTM
# lstm_model.fit(X_train_lstm, y_train, validation_data=(X_test_lstm, y_test), epochs=50, batch_size=64, callbacks=[early_stopping], verbose=1)
# y_pred_lstm = lstm_model.predict(X_test_lstm)
# mae_lstm = mean_absolute_error(y_test, y_pred_lstm)

# # **Scale data for ANN**
# scaler_X = StandardScaler()
# scaler_y = StandardScaler()

# X_train_scaled = scaler_X.fit_transform(X_train)
# X_test_scaled = scaler_X.transform(X_test)

# y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
# y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

# # **ANN Model**
# ann_model = Sequential([
#     Dense(128, activation='relu', input_dim=X_train_scaled.shape[1]),
#     Dropout(0.3),
#     Dense(64, activation='relu'),
#     Dropout(0.3),
#     Dense(1, activation='linear')
# ])

# ann_model.compile(optimizer=RMSprop(learning_rate=0.001), loss='mse', metrics=['mae'])
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# # Train ANN
# ann_model.fit(X_train_scaled, y_train_scaled, validation_data=(X_test_scaled, y_test_scaled), epochs=50, batch_size=64, callbacks=[early_stopping], verbose=1)
# y_pred_ann_scaled = ann_model.predict(X_test_scaled)

# # Inverse transform predictions and true values
# y_pred_ann = scaler_y.inverse_transform(y_pred_ann_scaled)
# y_test_original = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1))

# # Calculate MAE on original scale
# mae_ann = mean_absolute_error(y_test_original, y_pred_ann)

# # **Random Forest Model**
# rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
# rf_model.fit(X_train, y_train.ravel())
# y_pred_rf = rf_model.predict(X_test)
# mae_rf = mean_absolute_error(y_test, y_pred_rf)

# # **XGBoost Model**
# xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42)
# xgb_model.fit(X_train, y_train)
# y_pred_xgb = xgb_model.predict(X_test)
# mae_xgb = mean_absolute_error(y_test, y_pred_xgb)

# # **MAE Results**
# print(f"MAE for Dummy Model (Baseline): {mae_dummy}")
# print(f"MAE for LSTM: {mae_lstm}")
# print(f"MAE for ANN: {mae_ann}")
# print(f"MAE for Random Forest: {mae_rf}")
# print(f"MAE for XGBoost: {mae_xgb}")

# # Parity Plot Function
# def parity_plot(y_true, y_pred, model_name):
#     plt.figure(figsize=(6, 6))
#     plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='k')
#     plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--', linewidth=2)
#     plt.title(f'Parity Plot for {model_name}')
#     plt.xlabel('Actual Values')
#     plt.ylabel('Predicted Values')
#     plt.grid(True)
#     plt.show()

# # Generate Parity Plots
# parity_plot(y_test, y_pred_dummy, 'Dummy')
# parity_plot(y_test, y_pred_lstm, 'LSTM')
# parity_plot(y_test_original, y_pred_ann, 'ANN')
# parity_plot(y_test, y_pred_rf, 'Random Forest')
# parity_plot(y_test, y_pred_xgb, 'XGBoost')

import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from keras.layers import Input, LSTM, Dense, LeakyReLU, Dropout
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
import xgboost as xgb
from CalculateFeature import *
from get_data import *

# **Model Configuration**
model_name = 'BZT_Model'
target = get_model_target(model_name)[0]
state_variables_raw = get_state_variables(model_name)
positive_controls_var_raw = get_positive_controls(model_name)
negative_controls_var_raw = get_negative_controls(model_name)
all_variables_raw = get_all_variables(model_name)

# **Load and preprocess data**
raw_data = get_raw_data(all_variables_raw)

# **Remove duplicates from raw data**
raw_data = raw_data.drop_duplicates()

# Print raw data head after removing duplicates
print("\n=== Raw Data After Removing Duplicates (First 5 Rows) ===")
print(raw_data.head())

# Apply feature calculations for each state variable
for state_var in state_variables_raw:
    calculate_shifted_EMA(raw_data, state_var, BP=5, span=15)

# Apply feature calculations for positive and negative controls
for pos_control in positive_controls_var_raw:
    calculate_TS_Feature(raw_data, pos_control, BP=5, span=15)
for neg_control in negative_controls_var_raw:
    calculate_TS_Feature(raw_data, neg_control, BP=5, span=15)

# Apply EMA calculation for the target variable
calculate_EMA(raw_data, target, span=15)

# **Create final dataset with features and target variable**
state_vars = [var + '_EMA_Shifted' for var in state_variables_raw]
control_vars = [var + '_AUC' for var in positive_controls_var_raw + negative_controls_var_raw]
target_var = target + '_EMA'

# Create the final dataset by combining input features and target variable
final_data = raw_data[state_vars + control_vars + [target_var]].dropna()

# Print final data head
print("\n=== Final Data (First 5 Rows) ===")
print(final_data.head())

# **Split data into training and testing sets**
train_rows = int(len(final_data) * 0.8)
X_train = final_data[state_vars + control_vars].iloc[:train_rows].values
y_train = final_data[target_var].iloc[:train_rows].values
X_test = final_data[state_vars + control_vars].iloc[train_rows:].values
y_test = final_data[target_var].iloc[train_rows:].values

# **Dummy Model for Baseline Comparison**
dummy_model = DummyRegressor(strategy='mean')
dummy_model.fit(X_train, y_train)
y_pred_dummy = dummy_model.predict(X_test)
mae_dummy = mean_absolute_error(y_test, y_pred_dummy)

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

# **Scale data for ANN**
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

# **ANN Model**
ann_model = Sequential([
    Dense(128, activation='relu', input_dim=X_train_scaled.shape[1]),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='linear')
])

ann_model.compile(optimizer=RMSprop(learning_rate=0.001), loss='mse', metrics=['mae'])
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train ANN
ann_model.fit(X_train_scaled, y_train_scaled, validation_data=(X_test_scaled, y_test_scaled), epochs=50, batch_size=64, callbacks=[early_stopping], verbose=1)
y_pred_ann_scaled = ann_model.predict(X_test_scaled)

# Inverse transform predictions and true values
y_pred_ann = scaler_y.inverse_transform(y_pred_ann_scaled)
y_test_original = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1))

# Calculate MAE on original scale
mae_ann = mean_absolute_error(y_test_original, y_pred_ann)

# **Random Forest Model**
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train.ravel())
y_pred_rf = rf_model.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)

# **XGBoost Model**
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)

# **MAE Results**
print(f"MAE for Dummy Model (Baseline): {mae_dummy}")
print(f"MAE for LSTM: {mae_lstm}")
print(f"MAE for ANN: {mae_ann}")
print(f"MAE for Random Forest: {mae_rf}")
print(f"MAE for XGBoost: {mae_xgb}")

# Parity Plot Function
def parity_plot(y_true, y_pred, model_name):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='k')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--', linewidth=2)
    plt.title(f'Parity Plot for {model_name}')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    plt.show()

# Generate Parity Plots
parity_plot(y_test, y_pred_dummy, 'Dummy')
parity_plot(y_test, y_pred_lstm, 'LSTM')
parity_plot(y_test_original, y_pred_ann, 'ANN')
parity_plot(y_test, y_pred_rf, 'Random Forest')
parity_plot(y_test, y_pred_xgb, 'XGBoost')
