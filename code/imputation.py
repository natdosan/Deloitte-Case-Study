import pandas as pd
import numpy as np
import dask.dataframe as dd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def check_missing(x):
    """
    Checks whether or not a value in a series ends with a certain number

    Parameters
    ----------
    x : int
        value from series
    
    Returns
    -------
    np.Nan or x : True if it does end with the number otherwise false
    """
    str_x = str(x)
    if str_x.endswith(('91', '93', '94', '97', '98', '99', '.0')):
        return pd.NA
    else: 
        return x
    
def impute_na_with_random(series):
    """
    Replaces NA/NaN values in a pandas Series with samples drawn from the 
    non-missing values of that series.

    Parameters
    ----------
    series : pandas Series
        row from X

    Returns
    -------
    pandas Series : The input series with NA/NaN values replaced 
                    by sampled non-missing values.
    """
    mask = series.isna()
    samples = series[~mask].sample(mask.sum(), replace=True)
    samples.index = series[mask].index
    series[mask] = samples
    return series

def convert_numeric(df):
    """
    Convert all columns of DataFrame from object to int or float if possible.

    Parameters
    ----------
    df : pandas DataFrame
        Input DataFrame.

    Returns
    -------
    pandas DataFrame : DataFrame with columns converted to numeric types.
    """
    df = df.apply(lambda series: pd.to_numeric(series, errors='coerce'))
    return df