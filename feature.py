# import pandas as pd   
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# from scipy.stats import zscore
# from sqlalchemy import create_engine, text
# from genric import *  # Assuming this imports your custom functions

# # Load the model and get the associated data
# model_name = 'Kiln_Torque_Model'
# target = get_model_target(model_name)
# state_variables_raw = get_state_variables(model_name)
# positive_controls_var_raw = get_positive_controls(model_name)
# negative_controls_var_raw = get_negative_controls(model_name)
# all_variables_raw = get_all_variables(model_name)
# raw_data = get_raw_data(all_variables_raw)

# # Create small subsets of data
# def create_subsets(df, n_subsets=2):
#     subset_size = len(df) // n_subsets
#     subsets = [df[i*subset_size:(i+1)*subset_size] for i in range(n_subsets)]
#     return subsets

# # Filter out shutdown values
# def remove_shutdown_values(df, column, low=-10, high=50):
#     return df.loc[(df[column] < low) | (df[column] > high)]

# # Plot filtered data
# def plot_filtered_data(df, state_column, control_variable):
#     fig = make_subplots()
#     fig.add_trace(go.Scatter(x=df.index, y=df[control_variable], mode='lines', name=control_variable))
#     fig.add_trace(go.Scatter(x=df.index, y=df[state_column], mode='lines', name=state_column))
#     fig.update_layout(title="Filtered Data", xaxis_title="Time", yaxis_title="Value")
#     fig.show()

# # Calculate EMA with a vectorized approach
# def calculate_EMA(df, column, span=15):
#     df[f'{column}_EMA'] = df[column].ewm(span=span, adjust=False).mean()
#     return df

# # Calculate rolling AUC
# def calculate_rolling_auc(df, column, window=15):
#     return df[column].rolling(window=window).apply(lambda x: np.trapz(x), raw=True)

# # Calculate correlation efficiently with shifted EMA
# def calculate_correlation(BP, BP0, data, target_data, control_variable):
#     # Shift the EMA for the control variable and target data based on BP and BP0
#     shifted_control = data[f'{control_variable}_AUC'].shift(BP0).fillna(0)
#     shifted_target = target_data.shift(BP).fillna(0)
    
#     # Calculate correlation between shifted control variable and shifted target data
#     corr = np.corrcoef(shifted_control, shifted_target)[0, 1]
#     return {'Variable': control_variable, 'Correlation': corr, 'BP0': BP0, 'BP': BP}

# # Plot correlation heatmap
# def plot_correlation_heatmap(corr_df, control_variable, state_column):
#     if corr_df.empty:
#         print("No data to plot.")
#         return
#     heatmap_data = corr_df.pivot(index="BP", columns="BP0", values="Correlation")
#     plt.figure(figsize=(14, 10))  # Increased size for clarity with larger data
#     sns.heatmap(
#         heatmap_data, annot=True, fmt=".2f", cmap='coolwarm',
#         cbar_kws={'label': 'Correlation Coefficient'}, annot_kws={"size": 8}
#     )
#     plt.title(f'Heatmap of Correlation between {control_variable} and {state_column}', fontsize=14)
#     plt.xlabel('BP0', fontsize=12)
#     plt.ylabel('BP', fontsize=12)
#     plt.xticks(rotation=45, fontsize=8)
#     plt.yticks(rotation=0, fontsize=8)
#     plt.tight_layout()
#     plt.show()

# # Main process with optimized steps
# def main():
#     # Define variables
#     state_column = 'Kiln_Drive_Current'  # Replace with actual state variable column
#     selected_control_variable = 'Kiln_Feed_PV'  # Replace with actual control variable

#     # Create small data subsets for faster initial testing
#     data = get_raw_data(all_variables_raw)
#     data_subsets = create_subsets(data, n_subsets=5)
    
#     # Process each subset individually
#     for i, subset in enumerate(data_subsets):
#         print(f"Processing subset {i+1}/{len(data_subsets)}")
        
#         # Remove shutdown values
#         subset = remove_shutdown_values(subset, state_column)
        
#         # Calculate EMA and AUC
#         for col in [state_column, selected_control_variable]:
#             subset = calculate_EMA(subset, col, span=15)
#         subset[f'{selected_control_variable}_AUC'] = calculate_rolling_auc(subset, f'{selected_control_variable}_EMA', window=15)
        
#         # Prepare target data for shifted calculations
#         subset[f'{state_column}_EMA'] = subset[state_column].ewm(span=15, adjust=False).mean()
#         target_data = subset[f'{state_column}_EMA']
        
#         # Calculate correlations with varying BP and BP0
#         results = [
#             calculate_correlation(BP, BP0, subset, target_data, selected_control_variable)
#             for BP in range(1, 30)  # Expanded BP range
#             for BP0 in range(BP, 30)
#         ]
        
#         # Convert results to DataFrame and plot heatmap
#         corr_df = pd.DataFrame(results)
#         if not corr_df.empty:
#             plot_correlation_heatmap(corr_df, selected_control_variable, state_column)

# if __name__ == "__main__":
#     main()


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# from scipy.stats import zscore
# from sqlalchemy import create_engine, text
# from genric import *  # Assuming this imports your custom functions

# # Load model and associated data
# model_name = 'Kiln_Torque_Model'
# target = get_model_target(model_name)
# state_variables_raw = get_state_variables(model_name)
# positive_controls_var_raw = get_positive_controls(model_name)
# negative_controls_var_raw = get_negative_controls(model_name)
# all_variables_raw = get_all_variables(model_name)
# raw_data = get_raw_data(all_variables_raw)

# # Set state and control variables based on the model data
# state_column = state_variables_raw[0] if state_variables_raw else None

# # Create small subsets of data
# def create_subsets(df, n_subsets=2):
#     subset_size = len(df) // n_subsets
#     subsets = [df[i * subset_size:(i + 1) * subset_size] for i in range(n_subsets)]
#     return subsets

# # Filter out shutdown values
# def remove_shutdown_values(df, column, low=-10, high=50):
#     return df.loc[(df[column] < low) | (df[column] > high)]

# # Plot filtered data
# def plot_filtered_data(df, state_column, control_variable):
#     fig = make_subplots()
#     fig.add_trace(go.Scatter(x=df.index, y=df[control_variable], mode='lines', name=control_variable))
#     fig.add_trace(go.Scatter(x=df.index, y=df[state_column], mode='lines', name=state_column))
#     fig.update_layout(title="Filtered Data", xaxis_title="Time", yaxis_title="Value")
#     fig.show()

# # Calculate EMA with a vectorized approach
# def calculate_EMA(df, column, span=15):
#     df[f'{column}_EMA'] = df[column].ewm(span=span, adjust=False).mean()
#     return df

# # Calculate rolling AUC
# def calculate_rolling_auc(df, column, window=15):
#     return df[column].rolling(window=window).apply(lambda x: np.trapz(x), raw=True)

# # Optimized correlation calculation with fixed BP and varying BP0
# def calculate_correlation_for_BP0_range(data, target_data, control_variable, BP=5, max_BP0=30):
#     correlations = []
#     # Vectorize shifts for each BP0 value
#     for BP0 in range(1, max_BP0):
#         shifted_control = data[f'{control_variable}_AUC'].shift(BP0).fillna(0)
#         shifted_target = target_data.shift(BP).fillna(0)
#         corr = np.corrcoef(shifted_control, shifted_target)[0, 1]
#         correlations.append({'Variable': control_variable, 'Correlation': corr, 'BP0': BP0, 'BP': BP})
#     return correlations

# # Plot correlation heatmap
# def plot_correlation_heatmap(corr_df, control_variable, state_column):
#     if corr_df.empty:
#         print("No data to plot.")
#         return
#     heatmap_data = corr_df.pivot(index="BP", columns="BP0", values="Correlation")
#     plt.figure(figsize=(14, 10))
#     sns.heatmap(
#         heatmap_data, annot=True, fmt=".2f", cmap='coolwarm',
#         cbar_kws={'label': 'Correlation Coefficient'}, annot_kws={"size": 8}
#     )
#     plt.title(f'Heatmap of Correlation between {control_variable} and {state_column}', fontsize=14)
#     plt.xlabel('BP0', fontsize=12)
#     plt.ylabel('BP', fontsize=12)
#     plt.xticks(rotation=45, fontsize=8)
#     plt.yticks(rotation=0, fontsize=8)
#     plt.tight_layout()
#     plt.show()

# # Main process with optimized steps and multiple control variables
# def main():
#     # Ensure variables are properly set
#     if not (state_column and positive_controls_var_raw):
#         print("State or control variables are not defined. Please check the model data.")
#         return

#     # Create small data subsets for faster initial testing
#     data = get_raw_data(all_variables_raw)
#     data_subsets = create_subsets(data, n_subsets=5)
    
#     # Process each subset for each control variable
#     for control_variable in positive_controls_var_raw:
#         print(f"Processing for control variable: {control_variable}")
        
#         for i, subset in enumerate(data_subsets):
#             print(f"Processing subset {i+1}/{len(data_subsets)} for {control_variable}")
            
#             # Remove shutdown values
#             subset = remove_shutdown_values(subset, state_column)
            
#             # Calculate EMA and AUC for state and control variable
#             for col in [state_column, control_variable]:
#                 subset = calculate_EMA(subset, col, span=15)
#             subset[f'{control_variable}_AUC'] = calculate_rolling_auc(subset, f'{control_variable}_EMA', window=15)
            
#             # Prepare target data for shifted calculations
#             subset[f'{state_column}_EMA'] = subset[state_column].ewm(span=15, adjust=False).mean()
#             target_data = subset[f'{state_column}_EMA']
            
#             # Calculate correlations for varying BP0 and fixed BP
#             results = calculate_correlation_for_BP0_range(subset, target_data, control_variable, BP=5, max_BP0=30)
            
#             # Convert results to DataFrame and plot heatmap
#             corr_df = pd.DataFrame(results)
#             if not corr_df.empty:
#                 plot_correlation_heatmap(corr_df, control_variable, state_column)

# if __name__ == "__main__":
#     main()

# import pandas as pd    
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# from scipy.stats import zscore
# from sqlalchemy import create_engine, text
# from genric import *  # Assuming this imports your custom functions

# # Load the model and get the associated data
# model_name = 'Kiln_Torque_Model'
# target = get_model_target(model_name)
# state_variables_raw = get_state_variables(model_name)
# positive_controls_var_raw = get_positive_controls(model_name)
# negative_controls_var_raw = get_negative_controls(model_name)
# all_variables_raw = get_all_variables(model_name)
# raw_data = get_raw_data(all_variables_raw)

# # Choose state and control variables based on the model data
# state_column = state_variables_raw[0] if state_variables_raw else None
# positive_control_variable = positive_controls_var_raw[0] if positive_controls_var_raw else None
# negative_control_variable = negative_controls_var_raw[0] if negative_controls_var_raw else None

# # Create small subsets of data
# def create_subsets(df, n_subsets=2):
#     subset_size = len(df) // n_subsets
#     subsets = [df[i*subset_size:(i+1)*subset_size] for i in range(n_subsets)]
#     return subsets

# # Filter out shutdown values
# def remove_shutdown_values(df, column, low=-10, high=50):
#     return df.loc[(df[column] < low) | (df[column] > high)]

# # Plot filtered data
# def plot_filtered_data(df, state_column, control_variable):
#     fig = make_subplots()
#     fig.add_trace(go.Scatter(x=df.index, y=df[control_variable], mode='lines', name=control_variable))
#     fig.add_trace(go.Scatter(x=df.index, y=df[state_column], mode='lines', name=state_column))
#     fig.update_layout(title="Filtered Data", xaxis_title="Time", yaxis_title="Value")
#     fig.show()

# # Calculate EMA with a vectorized approach
# def calculate_EMA(df, column, span=15):
#     df[f'{column}_EMA'] = df[column].ewm(span=span, adjust=False).mean()
#     return df

# # Calculate rolling AUC
# def calculate_rolling_auc(df, column, window=15):
#     return df[column].rolling(window=window).apply(lambda x: np.trapzoid(x), raw=True)

# # Calculate correlation efficiently with shifted EMA
# def calculate_correlation(BP, BP0, data, target_data, control_variable):
#     shifted_control = data[f'{control_variable}_AUC'].shift(BP0).fillna(0)
#     shifted_target = target_data.shift(BP).fillna(0)
#     corr = np.corrcoef(shifted_control, shifted_target)[0, 1]
#     return {'Variable': control_variable, 'Correlation': corr, 'BP0': BP0, 'BP': BP}

# # Plot correlation heatmap
# def plot_correlation_heatmap(corr_df, control_variable, state_column):
#     if corr_df.empty:
#         print("No data to plot.")
#         return
#     heatmap_data = corr_df.pivot(index="BP", columns="BP0", values="Correlation")
#     plt.figure(figsize=(14, 10))
#     sns.heatmap(
#         heatmap_data, annot=True, fmt=".2f", cmap='coolwarm',
#         cbar_kws={'label': 'Correlation Coefficient'}, annot_kws={"size": 8}
#     )
#     plt.title(f'Heatmap of Correlation between {control_variable} and {state_column}', fontsize=14)
#     plt.xlabel('BP0', fontsize=12)
#     plt.ylabel('BP', fontsize=12)
#     plt.xticks(rotation=45, fontsize=8)
#     plt.yticks(rotation=0, fontsize=8)
#     plt.tight_layout()
#     plt.show()

# # Main process with optimized steps
# def main():
#     # Ensure variables are properly set
#     if not (state_column and positive_control_variable):
#         print("State or control variables are not defined. Please check the model data.")
#         return

#     # Create small data subsets for faster initial testing
#     data = get_raw_data(all_variables_raw)
#     data_subsets = create_subsets(data, n_subsets=5)
    
#     # Process each subset individually
#     for i, subset in enumerate(data_subsets):
#         print(f"Processing subset {i+1}/{len(data_subsets)}")
        
#         # Remove shutdown values
#         subset = remove_shutdown_values(subset, state_column)
        
#         # Calculate EMA and AUC
#         for col in [state_column, positive_control_variable]:
#             subset = calculate_EMA(subset, col, span=15)
#         subset[f'{positive_control_variable}_AUC'] = calculate_rolling_auc(subset, f'{positive_control_variable}_EMA', window=15)
        
#         # Prepare target data for shifted calculations
#         subset[f'{state_column}_EMA'] = subset[state_column].ewm(span=15, adjust=False).mean()
#         target_data = subset[f'{state_column}_EMA']
        
#         # Calculate correlations with varying BP and BP0
#         results = [
#             calculate_correlation(BP, BP0, subset, target_data, positive_control_variable)
#             for BP in range(1, 30)
#             for BP0 in range(BP, 30)
#         ]
        
#         # Convert results to DataFrame and plot heatmap
#         corr_df = pd.DataFrame(results)
#         if not corr_df.empty:
#             plot_correlation_heatmap(corr_df, positive_control_variable, state_column)

# if __name__ == "__main__":
#     main()

# import pandas as pd     
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# from scipy.stats import zscore
# from sqlalchemy import create_engine, text
# from genric import *  # Assuming this imports your custom functions

# # Load the model and get the associated data
# model_name = 'Kiln_Torque_Model'
# target = get_model_target(model_name)
# state_variables_raw = get_state_variables(model_name)
# positive_controls_var_raw = get_positive_controls(model_name)
# negative_controls_var_raw = get_negative_controls(model_name)
# all_variables_raw = get_all_variables(model_name)
# raw_data = get_raw_data(all_variables_raw)

# # Choose state and control variables based on the model data
# state_column = state_variables_raw[0] if state_variables_raw else None
# selected_control_variable = positive_controls_var_raw[0] if positive_controls_var_raw else None

# # Function to calculate EMA
# def calculate_EMA(df, column, span=15):
#     df[column + '_EMA'] = df[column].ewm(span=span, adjust=False).mean()
#     return df

# # Function to calculate shifted EMA
# def calculate_shifted_EMA(df, column, span=15, shift_periods=1):
#     df[column + f'_EMA_shifted_{shift_periods}'] = df[column].ewm(span=span, adjust=False).mean().shift(shift_periods)
#     return df

# # Function to create small subsets of data
# def create_subsets(df, n_subsets=2):
#     subset_size = len(df) // n_subsets
#     subsets = [df[i*subset_size:(i+1)*subset_size] for i in range(n_subsets)]
#     return subsets

# # Function to remove shutdown values
# def remove_shutdown_values(df, column, low=-10, high=50):
#     return df.loc[(df[column] < low) | (df[column] > high)]

# # Plot filtered data
# def plot_filtered_data(df, state_column, control_variable):
#     fig = make_subplots()
#     fig.add_trace(go.Scatter(x=df.index, y=df[control_variable], mode='lines', name=control_variable))
#     fig.add_trace(go.Scatter(x=df.index, y=df[state_column], mode='lines', name=state_column))
#     fig.update_layout(title="Filtered Data", xaxis_title="Time", yaxis_title="Value")
#     fig.show()

# # Calculate rolling AUC
# def calculate_rolling_auc(df, column, window=15):
#     return df[column].rolling(window=window).apply(lambda x: np.trapz(x), raw=True)

# # Calculate correlation with rolling and shifted EMA
# def calculate_correlation(BP, BP0, data, target_data, control_data):
#     shifted_control_data = control_data.shift(BP0).fillna(0)
#     shifted_target_data = target_data.shift(BP0).fillna(0)
    
#     # Apply a rolling window based on BP
#     if BP > 1:
#         rolled_control_data = shifted_control_data.rolling(window=BP).mean().fillna(0)
#         rolled_target_data = shifted_target_data.rolling(window=BP).mean().fillna(0)
#     else:
#         rolled_control_data = shifted_control_data
#         rolled_target_data = shifted_target_data

#     # Calculate the correlation
#     corr = np.corrcoef(rolled_control_data, rolled_target_data)[0, 1]
#     return {'Variable': selected_control_variable, 'Correlation': corr, 'BP0': BP0, 'BP': BP}

# # Plot correlation heatmap
# def plot_correlation_heatmap(corr_df, control_variable, state_column):
#     if corr_df.empty:
#         print("No data to plot.")
#         return
#     heatmap_data = corr_df.pivot(index="BP", columns="BP0", values="Correlation")
#     plt.figure(figsize=(14, 10))
#     sns.heatmap(
#         heatmap_data, annot=True, fmt=".2f", cmap='coolwarm',
#         cbar_kws={'label': 'Correlation Coefficient'}, annot_kws={"size": 8}
#     )
#     plt.title(f'Heatmap of Correlation between {control_variable} and {state_column}', fontsize=14)
#     plt.xlabel('BP0', fontsize=12)
#     plt.ylabel('BP', fontsize=12)
#     plt.xticks(rotation=45, fontsize=8)
#     plt.yticks(rotation=0, fontsize=8)
#     plt.tight_layout()
#     plt.show()

# # Main process with optimized steps
# def main():
#     # Ensure variables are properly set
#     if not (state_column and selected_control_variable):
#         print("State or control variables are not defined. Please check the model data.")
#         return

#     # Create small data subsets for faster initial testing
#     data = get_raw_data(all_variables_raw)
#     data_subsets = create_subsets(data, n_subsets=5)
    
#     # Process each subset individually
#     for i, subset in enumerate(data_subsets):
#         print(f"Processing subset {i+1}/{len(data_subsets)}")
        
#         # Remove shutdown values
#         subset = remove_shutdown_values(subset, state_column)
        
#         # Calculate EMA and shifted EMA for state and control variables
#         for col in [state_column, selected_control_variable]:
#             subset = calculate_EMA(subset, col, span=15)
#             subset = calculate_shifted_EMA(subset, col, span=15, shift_periods=5)
        
#         # Calculate AUC on the control variable's EMA
#         subset[f'{selected_control_variable}_AUC'] = calculate_rolling_auc(subset, f'{selected_control_variable}_EMA', window=15)
        
#         # Prepare target data for shifted calculations
#         target_data = subset[f'{state_column}_EMA_shifted_5']
#         control_data = subset[f'{selected_control_variable}_EMA_shifted_5']
        
#         # Calculate correlations with varying BP and BP0
#         results = [
#             calculate_correlation(BP, BP0, subset, target_data, control_data)
#             for BP in range(1, 30)
#             for BP0 in range(BP, 30)
#         ]
        
#         # Convert results to DataFrame and plot heatmap
#         corr_df = pd.DataFrame(results)
#         if not corr_df.empty:
#             plot_correlation_heatmap(corr_df, selected_control_variable, state_column)

# if __name__ == "__main__":
#     main()

import pandas as pd      
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import zscore
from sqlalchemy import create_engine, text
from genric import *  # Assuming this imports your custom functions

# Load the model and get the associated data
model_name = 'Kiln_Torque_Model'
target = get_model_target(model_name)
state_variables_raw = get_state_variables(model_name)
positive_controls_var_raw = get_positive_controls(model_name)
negative_controls_var_raw = get_negative_controls(model_name)
all_variables_raw = get_all_variables(model_name)
raw_data = get_raw_data(all_variables_raw)

# Ensure all variables are available
state_column = state_variables_raw[0] if state_variables_raw else None
positive_control_variables = positive_controls_var_raw if positive_controls_var_raw else []
negative_control_variables = negative_controls_var_raw if negative_controls_var_raw else []

# Function to calculate EMA
def calculate_EMA(df, column, span=15):
    df[column + '_EMA'] = df[column].ewm(span=span, adjust=False).mean()
    return df

# Function to calculate shifted EMA
def calculate_shifted_EMA(df, column, span=15, shift_periods=1):
    df[column + f'_EMA_shifted_{shift_periods}'] = df[column].ewm(span=span, adjust=False).mean().shift(shift_periods)
    return df

# Function to create small subsets of data
def create_subsets(df, n_subsets=2):
    subset_size = len(df) // n_subsets
    subsets = [df[i*subset_size:(i+1)*subset_size] for i in range(n_subsets)]
    return subsets

# Function to remove shutdown values
def remove_shutdown_values(df, column, low=-10, high=50):
    return df.loc[(df[column] < low) | (df[column] > high)]

# Calculate correlation with rolling and shifted EMA
def calculate_correlation(BP, BP0, data, target_data, control_data):
    shifted_control_data = control_data.shift(BP0).fillna(0)
    shifted_target_data = target_data.shift(BP0).fillna(0)
    
    # Apply a rolling window based on BP
    if BP > 1:
        rolled_control_data = shifted_control_data.rolling(window=BP).mean().fillna(0)
        rolled_target_data = shifted_target_data.rolling(window=BP).mean().fillna(0)
    else:
        rolled_control_data = shifted_control_data
        rolled_target_data = shifted_target_data

    # Calculate the correlation
    corr = np.corrcoef(rolled_control_data, rolled_target_data)[0, 1]
    return {'Variable': control_data.name, 'Correlation': corr, 'BP0': BP0, 'BP': BP}

# Plot correlation heatmap for each control variable
def plot_correlation_heatmap(corr_df, control_variable, state_column):
    if corr_df.empty:
        print("No data to plot.")
        return
    heatmap_data = corr_df.pivot(index="BP", columns="BP0", values="Correlation")
    plt.figure(figsize=(14, 10))
    sns.heatmap(
        heatmap_data, annot=True, fmt=".2f", cmap='coolwarm',
        cbar_kws={'label': 'Correlation Coefficient'}, annot_kws={"size": 8}
    )
    plt.title(f'Heatmap of Correlation between {control_variable} and {state_column}', fontsize=14)
    plt.xlabel('BP0', fontsize=12)
    plt.ylabel('BP', fontsize=12)
    plt.xticks(rotation=45, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.show()

# Main process with optimized steps for plotting multiple heatmaps
def main():
    # Ensure variables are properly set
    if not (state_column and positive_control_variables):
        print("State or control variables are not defined. Please check the model data.")
        return

    # Create small data subsets for faster initial testing
    data = get_raw_data(all_variables_raw)
    data_subsets = create_subsets(data, n_subsets=5)
    
    # Process each subset individually
    for i, subset in enumerate(data_subsets):
        print(f"Processing subset {i+1}/{len(data_subsets)}")
        
        # Remove shutdown values
        subset = remove_shutdown_values(subset, state_column)
        
        # Calculate EMA and shifted EMA for state and control variables
        for col in [state_column] + positive_control_variables + negative_control_variables:
            subset = calculate_EMA(subset, col, span=15)
            subset = calculate_shifted_EMA(subset, col, span=15, shift_periods=5)
        
        # Prepare target data for shifted calculations
        target_data = subset[f'{state_column}_EMA_shifted_5']
        
        # Loop through each positive and negative control variable
        for control_variable in positive_control_variables + negative_control_variables:
            control_data = subset[f'{control_variable}_EMA_shifted_5']
            
            # Calculate correlations with varying BP and BP0 for each control variable
            results = [
                calculate_correlation(BP, BP0, subset, target_data, control_data)
                for BP in range(1, 30)
                for BP0 in range(BP, 30)
            ]
            
            # Convert results to DataFrame
            corr_df = pd.DataFrame(results)
            if not corr_df.empty:
                # Plot heatmap for each control variable
                plot_correlation_heatmap(corr_df, control_variable, state_column)

if __name__ == "__main__":
    main()
# import pandas as pd       
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# from scipy.stats import zscore
# from sqlalchemy import create_engine, text
# from genric import *  # Assuming this imports your custom functions

# # Load the model and get the associated data
# model_name = 'Kiln_Torque_Model'
# target = get_model_target(model_name)
# state_variables_raw = get_state_variables(model_name)
# positive_controls_var_raw = get_positive_controls(model_name)
# negative_controls_var_raw = get_negative_controls(model_name)
# all_variables_raw = get_all_variables(model_name)
# raw_data = get_raw_data(all_variables_raw)

# # Ensure all variables are available
# state_column = state_variables_raw[0] if state_variables_raw else None
# positive_control_variables = positive_controls_var_raw if positive_controls_var_raw else []
# negative_control_variables = negative_controls_var_raw if negative_controls_var_raw else []

# # Function to calculate EMA
# def calculate_EMA(df, column, span=15):
#     df[column + '_EMA'] = df[column].ewm(span=span, adjust=False).mean()
#     return df

# # Function to calculate shifted EMA
# def calculate_shifted_EMA(df, column, span=15, shift_periods=1):
#     df[column + f'_EMA_shifted_{shift_periods}'] = df[column].ewm(span=span, adjust=False).mean().shift(shift_periods)
#     return df

# # Function to create small subsets of data
# def create_subsets(df, n_subsets=1):
#     subset_size = len(df) // n_subsets
#     subsets = [df[i*subset_size:(i+1)*subset_size] for i in range(n_subsets)]
#     return subsets

# # Function to remove shutdown values
# def remove_shutdown_values(df, column, low=-10, high=50):
#     return df.loc[(df[column] < low) | (df[column] > high)]

# # Calculate correlation with rolling and shifted EMA
# def calculate_correlation(BP, BP0, data, target_data, control_data):
#     shifted_control_data = control_data.shift(BP0).fillna(0)
#     shifted_target_data = target_data.shift(BP0).fillna(0)
    
#     # Apply a rolling window based on BP
#     if BP > 1:
#         rolled_control_data = shifted_control_data.rolling(window=BP).mean().fillna(0)
#         rolled_target_data = shifted_target_data.rolling(window=BP).mean().fillna(0)
#     else:
#         rolled_control_data = shifted_control_data
#         rolled_target_data = shifted_target_data

#     # Calculate the correlation
#     corr = np.corrcoef(rolled_control_data, rolled_target_data)[0, 1]
#     return {'Variable': control_data.name, 'Correlation': corr, 'BP0': BP0, 'BP': BP}

# # Plot correlation heatmap for each control variable
# def plot_correlation_heatmap(corr_df, control_variable, state_column):
#     if corr_df.empty:
#         print(f"No data to plot for {control_variable}.")
#         return
#     heatmap_data = corr_df.pivot(index="BP", columns="BP0", values="Correlation")
#     plt.figure(figsize=(14, 10))
#     sns.heatmap(
#         heatmap_data, annot=True, fmt=".2f", cmap='coolwarm',
#         cbar_kws={'label': 'Correlation Coefficient'}, annot_kws={"size": 8}
#     )
#     plt.title(f'Heatmap of Correlation between {control_variable} and {state_column}', fontsize=14)
#     plt.xlabel('BP0', fontsize=12)
#     plt.ylabel('BP', fontsize=12)
#     plt.xticks(rotation=45, fontsize=8)
#     plt.yticks(rotation=0, fontsize=8)
#     plt.tight_layout()
#     plt.show()

# # Main process with optimized steps for plotting multiple heatmaps
# def main():
#     # Ensure variables are properly set
#     if not (state_column and positive_control_variables):
#         print("State or control variables are not defined. Please check the model data.")
#         return

#     # Create small data subsets for faster initial testing
#     data = get_raw_data(all_variables_raw)
#     data_subsets = create_subsets(data, n_subsets=5)
    
#     # Process each subset individually
#     for i, subset in enumerate(data_subsets):
#         print(f"Processing subset {i+1}/{len(data_subsets)}")
        
#         # Remove shutdown values
#         subset = remove_shutdown_values(subset, state_column)
        
#         # Calculate EMA and shifted EMA for state and control variables
#         for col in [state_column] + positive_control_variables + negative_control_variables:
#             subset = calculate_EMA(subset, col, span=15)
#             subset = calculate_shifted_EMA(subset, col, span=15, shift_periods=5)
        
#         # Prepare target data for shifted calculations
#         target_data = subset[f'{state_column}_EMA_shifted_5']
        
#         # Loop through each positive and negative control variable
#         for control_variable in positive_control_variables + negative_control_variables:
#             control_data = subset[f'{control_variable}_EMA_shifted_5']
            
#             # Calculate correlations with varying BP and BP0 for each control variable
#             results = [
#                 calculate_correlation(BP, BP0, subset, target_data, control_data)
#                 for BP in range(1, 30)
#                 for BP0 in range(BP, 30)
#             ]
            
#             # Convert results to DataFrame
#             corr_df = pd.DataFrame(results)
#             if not corr_df.empty:
#                 # Plot heatmap for each control variable
#                 plot_correlation_heatmap(corr_df, control_variable, state_column)

# if __name__ == "__main__":
#     main()
