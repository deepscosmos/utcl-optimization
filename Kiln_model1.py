from genric import * 
from  features import *

model_name='Kiln_Torque_Model'
target=get_model_target(model_name)
state_variables_raw=get_state_variables(model_name)
positive_controls_var_raw=get_positive_controls(model_name)
negative_controls_var_raw=get_negative_controls(model_name)
all_variables_raw=get_all_variables(model_name)
raw_data=get_raw_data(all_variables_raw)
print(target)

for state_var in state_variables_raw:
    calculate_shifted_EMA(raw_data, state_var, BP=5, span=15)
for pos_control in positive_controls_var_raw:
    calculate_TS_Feature(raw_data, pos_control, BP=5, span=15)
for neg_control in negative_controls_var_raw:
    calculate_TS_Feature(raw_data, neg_control, BP=5, span=15)
calculate_EMA(raw_data, target[0], span=15)


State_vars=[]
for state_var in state_variables_raw:
    State_vars.append(state_var+'_EMA_Shifted')
for pos_control in positive_controls_var_raw:
    State_vars.append(pos_control+'_EMA_Shifted')
for neg_control in negative_controls_var_raw:
    State_vars.append(neg_control+'_EMA_Shifted')
neg_control_vars=[]
for neg_control in negative_controls_var_raw:
    neg_control_vars.append(neg_control+'_AUC')
pos_control_vars=[]
for pos_control in positive_controls_var_raw:
    pos_control_vars.append(pos_control+'_AUC')

target_vars=target[0]+'_EMA'


X_cols=State_vars+neg_control_vars+pos_control_vars
y_cols=target_vars

final_data=raw_data[X_cols+[y_cols]]
final_data.dropna(inplace=True)
train_rows=int(len(final_data)*0.8)

X_train=final_data[X_cols].iloc[:train_rows]
y_train=final_data[y_cols].iloc[:train_rows]
X_test=final_data[X_cols].iloc[train_rows:]
y_test=final_data[y_cols].iloc[train_rows:]

# Train a simple linear regression model and a dummy regressor
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.dummy import DummyRegressor

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(),
    "Neural Network": MLPRegressor(),
    "Dummy": DummyRegressor(),
}
results = []

for name, model in models.items():
    if name == "Dummy":
        model.fit(X_train, y_train-X_train[target[0]+'_EMA_Shifted'])
    else:
        model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = np.mean(np.abs(y_pred - y_test))
    results.append((name, mae))

# Print the results after the loop
for name, mae in results:
    print(f"{name} MAE: {mae}")

# # column names should appear in sorted order of their names
# raw_data = raw_data.reindex(sorted(raw_data.columns), axis=1)
# print(raw_data.head())


# # line plot of target variable and target variable EMA
# import matplotlib.pyplot as plt

# # Plot only the first 1000 points
# plt.figure(figsize=(15, 6))
# plt.plot(raw_data[target[0]].iloc[:1000], label='Target')
# plt.plot(raw_data[target[0]+'_EMA'].iloc[:1000], label='Target EMA')
# plt.legend()
# plt.show()