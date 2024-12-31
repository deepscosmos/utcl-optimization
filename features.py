import pandas as pd
import numpy as np

def calculate_EMA(df, column, span=10):
    """
    `Calculate smooth target using this function.`
    Calculate the exponentially smoothed value of a given column.
    
    Parameters:
    df (pd.DataFrame): DataFrame with datetime index.
    column (str): Column name to calculate EMA for.
    span (int): Span for exponential smoothing.
    
    Returns:
    pd.DataFrame: DataFrame with new feature added.
    """
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
    if BP < 0:
        raise ValueError("Backward prediction is not supported")
    df[column + '_EMA'] = df[column].ewm(span=span, adjust=False).mean()
    df[column + '_EMA_Shifted'] = df[column + '_EMA'].shift(BP, freq='min')
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
    if BP < 0:
        raise ValueError("Backward prediction is not supported")
    
    # Calculate exponentially smoothed value
    df[column + '_EMA'] = df[column].ewm(span=span, adjust=False).mean()
    df[column + '_EMA_Shifted'] = df[column + '_EMA'].shift(BP, freq='min')
    
    df[column+'_diff']=df[column]-df[column].shift(BP, freq='min')
    # Calculate area under the curve
    df[column + '_AUC'] = df[column+'_diff'].rolling(f'{BP}min').apply(lambda x: np.trapz(x, dx=1), raw=True)
    # Calculate mean area under the curve
    df[column + '_AUC'] = df[column + '_AUC']/BP
    df.drop(column+'_diff', axis=1, inplace=True)
    
    return df