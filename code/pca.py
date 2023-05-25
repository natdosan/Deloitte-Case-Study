import pandas as pd
import numpy as np
import dask.dataframe as dd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def pca(X, scaled_data, thres = 0.95):
    """
    Performs PCA on a dataset and prints out the components

    Parameters
    ----------
    X : pandas DataFrame
        data we want to perform PCA on
    scaled_data : pandas DataFrame
        standardized version of X
    thres : float
        % threshold for the variance

    Returns 
    -------
    None
    """
    pca = PCA()
    pca.fit(scaled_data)

    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    variance_threshold = thres # Set your desired threshold here (default 95%)
    n_components = np.argmax(cumulative_variance_ratio >= variance_threshold) + 1

    print(f'There are {n_components} components that explain the variance within the dataset')

    loadings = pca.components_
    absolute_loadings = np.abs(loadings)
    feature_importance = np.mean(absolute_loadings, axis=0)
    sorted_indices = np.argsort(feature_importance)[::-1]
    feature_names = list(X.columns)
    for i in sorted_indices:
        print(f"{feature_names[i]}: {feature_importance[i]}")

def create_correlation_matrix(X):
    """
    Creates a correlation matrix given a dataset, X

    Parameters
    ----------
    X : pandas DataFrame
        data we want to get corr mtx of

    Returns 
    -------
    correlation_with_sumflag : pandas Series
        correlation features with SUMFLAG column
    """
    correlation_matrix = X.corr()
    correlation_with_sumflag = correlation_matrix["SUMFLAG"]
    return correlation_with_sumflag