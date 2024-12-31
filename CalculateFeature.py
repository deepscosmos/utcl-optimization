# import pandas as pd
# import numpy as np

# def calculate_EMA(df, column, span=10):
#     """
#     `Calculate smooth target using this function.`
#     Calculate the exponentially smoothed value of a given column.
    
#     Parameters:
#     df (pd.DataFrame): DataFrame with datetime index.
#     column (str): Column name to calculate EMA for.
#     span (int): Span for exponential smoothing.
    
#     Returns:
#     pd.DataFrame: DataFrame with new feature added.
#     """
#     df[column + '_EMA'] = df[column].ewm(span=span, adjust=False).mean()
#     return df

# def calculate_shifted_EMA(df, column, BP, span=10):
#     """
#     Calculate the exponentially smoothed value of a given column at time (t-BP).
    
#     Parameters:
#     df (pd.DataFrame): DataFrame with datetime index.
#     column (str): Column name to calculate EMA for.
#     BP (int): Backward period in minutes.
#     span (int): Span for exponential smoothing.
    
#     Returns:
#     pd.DataFrame: DataFrame with new feature added.
#     """
#     if BP < 0:
#         raise ValueError("Backward prediction is not supported")
#     df[column + '_EMA'] = df[column].ewm(span=span, adjust=False).mean()
#     df[column + '_EMA_Shifted'] = df[column + '_EMA'].shift(BP, freq='min')
#     df.drop(column + '_EMA', axis=1, inplace=True)
#     return df

# def calculate_TS_Feature(df, column, BP, span=10):
#     """
#     Calculate two features for a given column:
#     1. Exponentially smoothed value of the column at time (t-BP).
#     2. Mean area under the curve of the raw column from (t-BP) to t.
    
#     Parameters:
#     df (pd.DataFrame): DataFrame with datetime index.
#     column (str): Column name to calculate features for.
#     BP (int): Backward period in minutes.
#     span (int): Span for exponential smoothing.
    
#     Returns:
#     pd.DataFrame: DataFrame with new features added.
#     """
#     if BP < 0:
#         raise ValueError("Backward prediction is not supported")
    
#     # Calculate exponentially smoothed value
#     df[column + '_EMA'] = df[column].ewm(span=span, adjust=False).mean()
#     df[column + '_EMA_Shifted'] = df[column + '_EMA'].shift(BP, freq='min')
    
#     df[column+'_diff']=df[column]-df[column].shift(BP, freq='min')
#     # Calculate area under the curve
#     df[column + '_AUC'] = df[column+'_diff'].rolling(f'{BP}min').apply(lambda x: np.trapz(x, dx=1), raw=True)
#     # Calculate mean area under the curve
#     df[column + '_AUC'] = df[column + '_AUC']/BP
#     df.drop(column+'_diff', axis=1, inplace=True)
    
#     return df

# import pandas as pd
# import numpy as np

# def calculate_EMA(df, column, span=10):
#     """
#     Calculate the exponentially smoothed value of a given column.
    
#     Parameters:
#     df (pd.DataFrame): DataFrame with datetime index.
#     column (str): Column name to calculate EMA for.
#     span (int): Span for exponential smoothing.
    
#     Returns:
#     pd.DataFrame: DataFrame with new feature added.
#     """
#     # Ensure the index is datetime
#     if not isinstance(df.index, pd.DatetimeIndex):
#         raise ValueError("DataFrame index must be a DatetimeIndex.")
    
#     # Calculate EMA
#     df[column + '_EMA'] = df[column].ewm(span=span, adjust=False).mean()
#     return df


# def calculate_shifted_EMA(df, column, BP, span=10):
#     """
#     Calculate the exponentially smoothed value of a given column at time (t-BP).
    
#     Parameters:
#     df (pd.DataFrame): DataFrame with datetime index.
#     column (str): Column name to calculate EMA for.
#     BP (int): Backward period in minutes.
#     span (int): Span for exponential smoothing.
    
#     Returns:
#     pd.DataFrame: DataFrame with new feature added.
#     """
#     # Ensure the index is datetime
#     if not isinstance(df.index, pd.DatetimeIndex):
#         raise ValueError("DataFrame index must be a DatetimeIndex.")
    
#     if BP < 0:
#         raise ValueError("Backward prediction is not supported.")
    
#     # Handle duplicate indices
#     if df.index.duplicated().sum() > 0:
#         print("Duplicate indices found. Removing duplicates...")
#         df = df[~df.index.duplicated(keep='first')]

#     # Calculate EMA
#     df[column + '_EMA'] = df[column].ewm(span=span, adjust=False).mean()
#     print(df.head())

#     # Shift EMA by BP minutes
#     df[column + '_EMA_Shifted'] = df[column + '_EMA'].shift(BP, freq='min')

#     # Drop the temporary EMA column
#     df.drop(column + '_EMA', axis=1, inplace=True)
#     return df


# def calculate_TS_Feature(df, column, BP, span=10):
#     """
#     Calculate two features for a given column:
#     1. Exponentially smoothed value of the column at time (t-BP).
#     2. Mean area under the curve of the raw column from (t-BP) to t.
    
#     Parameters:
#     df (pd.DataFrame): DataFrame with datetime index.
#     column (str): Column name to calculate features for.
#     BP (int): Backward period in minutes.
#     span (int): Span for exponential smoothing.
    
#     Returns:
#     pd.DataFrame: DataFrame with new features added.
#     """
#     # Ensure the index is datetime
#     if not isinstance(df.index, pd.DatetimeIndex):
#         raise ValueError("DataFrame index must be a DatetimeIndex.")
    
#     if BP < 0:
#         raise ValueError("Backward prediction is not supported.")
    
#     # Handle duplicate indices
#     if df.index.duplicated().sum() > 0:
#         print("Duplicate indices found. Removing duplicates...")
#         df = df[~df.index.duplicated(keep='first')]

#     # Calculate exponentially smoothed value
#     df[column + '_EMA'] = df[column].ewm(span=span, adjust=False).mean()
#     df[column + '_EMA_Shifted'] = df[column + '_EMA'].shift(BP, freq='min')

#     # Calculate the difference for area under the curve
#     df[column + '_diff'] = df[column] - df[column].shift(BP, freq='min')

#     # Calculate area under the curve using trapezoidal integration
#     df[column + '_AUC'] = (
#         df[column + '_diff']
#         .rolling(f'{BP}min')
#         .apply(lambda x: np.trapz(x, dx=1), raw=True)
#     )

#     # Calculate mean area under the curve
#     df[column + '_AUC'] = df[column + '_AUC'] / BP

#     # Clean up temporary columns
#     df.drop(column + '_diff', axis=1, inplace=True)
#     df.drop(column + '_EMA', axis=1, inplace=True)

#     return df

import pandas as pd
import numpy as np


def calculate_EMA(df, column, span=10):
    """
    Calculate the exponentially smoothed value of a given column.
    
    Parameters:
    df (pd.DataFrame): DataFrame with datetime index.
    column (str): Column name to calculate EMA for.
    span (int): Span for exponential smoothing.
    
    Returns:
    pd.DataFrame: DataFrame with new feature added.
    """
    # Ensure the index is datetime and sorted
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")
    df = df.sort_index()

    # Handle duplicate indices
    if df.index.duplicated().sum() > 0:
        print("Duplicate indices found. Removing duplicates...")
        df = df[~df.index.duplicated(keep='first')]

    # Calculate EMA
    df[column + '_EMA'] = df[column].ewm(span=span, adjust=False).mean()
    return df


def calculate_shifted_EMA(df, column, BP, span=10):
    """
    Calculate the exponentially smoothed value of a given column at time (t-BP).
    
    Parameters:
    df (pd.DataFrame): DataFrame with datetime index.
    column (str): Column name to calculate EMA for.
    BP (int): Backward period in minutes.
    span (int): Span for exponential smoothing.
    
    Returns:
    pd.DataFrame: DataFrame with new feature added.
    """
    # Ensure the index is datetime and sorted
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")
    df = df.sort_index()

    # Handle duplicate indices
    if df.index.duplicated().sum() > 0:
        print("Duplicate indices found. Removing duplicates...")
        df = df[~df.index.duplicated(keep='first')]

    # Calculate EMA
    df[column + '_EMA'] = df[column].ewm(span=span, adjust=False).mean()

    # Shift EMA by BP minutes
    df[column + '_EMA_Shifted'] = df[column + '_EMA'].shift(BP, freq='min')

    # Drop the temporary EMA column
    df.drop(column + '_EMA', axis=1, inplace=True)
    return df


def calculate_TS_Feature(df, column, BP, span=10):
    """
    Calculate two features for a given column:
    1. Exponentially smoothed value of the column at time (t-BP).
    2. Mean area under the curve of the raw column from (t-BP) to t.
    
    Parameters:
    df (pd.DataFrame): DataFrame with datetime index.
    column (str): Column name to calculate features for.
    BP (int): Backward period in minutes.
    span (int): Span for exponential smoothing.
    
    Returns:
    pd.DataFrame: DataFrame with new features added.
    """
    # Ensure the index is datetime and sorted
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")
    df = df.sort_index()

    # Handle duplicate indices
    if df.index.duplicated().sum() > 0:
        print("Duplicate indices found. Removing duplicates...")
        df = df[~df.index.duplicated(keep='first')]

    # Calculate exponentially smoothed value
    df[column + '_EMA'] = df[column].ewm(span=span, adjust=False).mean()
    df[column + '_EMA_Shifted'] = df[column + '_EMA'].shift(BP, freq='min')

    # Calculate the difference for area under the curve
    df[column + '_diff'] = df[column] - df[column].shift(BP, freq='min')

    # Calculate area under the curve using trapezoidal integration
    df[column + '_AUC'] = (
        df[column + '_diff']
        .rolling(f'{BP}min')
        .apply(lambda x: np.trapz(x, dx=1), raw=True)
    )

    # Calculate mean area under the curve
    df[column + '_AUC'] = df[column + '_AUC'] / BP

    # Clean up temporary columns
    df.drop(column + '_diff', axis=1, inplace=True)
    df.drop(column + '_EMA', axis=1, inplace=True)

    return df
