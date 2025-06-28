"""
Robust Covariance Estimation Methods

This module provides implementations of robust covariance estimation methods:
1. Ledoit-Wolf Shrinkage
2. Exponentially Weighted Moving Average (EWMA)
3. Combined EWMA with Shrinkage
"""

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def sample_covariance(returns, annualization=252):
    """
    Calculate simple sample covariance matrix
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Asset returns
    annualization : int
        Annualization factor (252 for daily returns)
        
    Returns:
    --------
    pd.DataFrame : Covariance matrix
    """
    return returns.cov() * annualization


def ledoit_wolf_covariance(returns, annualization=252):
    """
    Calculate covariance with Ledoit-Wolf shrinkage
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Asset returns
    annualization : int
        Annualization factor (252 for daily returns)
        
    Returns:
    --------
    pd.DataFrame : Covariance matrix with Ledoit-Wolf shrinkage applied
    """
    # Ensure we have non-empty returns without NaN values
    if returns.empty or returns.isnull().values.any():
        raise ValueError("Returns contain NaN values")
    
    assets = returns.columns
    lw = LedoitWolf().fit(returns.values)
    
    # Get the shrunk covariance matrix and convert to DataFrame
    cov = pd.DataFrame(lw.covariance_, index=assets, columns=assets) * annualization
    
    return cov


def ewma_covariance(returns, lambda_=0.94, annualization=252):
    """
    Calculate covariance using Exponentially Weighted Moving Average
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Asset returns
    lambda_ : float
        Decay factor (higher = more weight on older observations)
        Typical values: 0.94 (equiv to ~32-day half-life) for daily returns
    annualization : int
        Annualization factor (252 for daily returns)
        
    Returns:
    --------
    pd.DataFrame : EWMA covariance matrix
    """
    assets = returns.columns
    n = returns.shape[0]
    
    # Initialize weights for EWMA
    weights = np.zeros(n)
    for i in range(n):
        weights[i] = (1 - lambda_) * (lambda_ ** (n - 1 - i))
    
    # Normalize weights
    weights /= np.sum(weights)
    
    # Compute weighted covariance
    weighted_returns = returns - returns.mean()
    weighted_returns = weighted_returns * np.sqrt(weights[:, np.newaxis])
    ewma_cov = weighted_returns.T @ weighted_returns
    
    # Convert to DataFrame and annualize
    ewma_cov = pd.DataFrame(ewma_cov, index=assets, columns=assets) * annualization
    
    return ewma_cov


def ewma_correlation(returns, lambda_=0.94):
    """
    Calculate correlation using Exponentially Weighted Moving Average
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Asset returns
    lambda_ : float
        Decay factor
        
    Returns:
    --------
    pd.DataFrame : EWMA correlation matrix
    """
    # Get EWMA covariance (without annualization)
    cov = ewma_covariance(returns, lambda_=lambda_, annualization=1)
    
    # Compute correlation from covariance
    std = np.sqrt(np.diag(cov))
    corr = cov.copy()
    
    for i in range(len(std)):
        for j in range(len(std)):
            if std[i] > 0 and std[j] > 0:
                corr.iloc[i, j] = cov.iloc[i, j] / (std[i] * std[j])
            else:
                corr.iloc[i, j] = 0 if i != j else 1
                
    return corr


def combined_ewma_shrinkage(returns, lambda_=0.94, shrinkage_intensity=0.5, annualization=252):
    """
    Calculate covariance using EWMA with additional shrinkage applied
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Asset returns
    lambda_ : float
        EWMA decay factor
    shrinkage_intensity : float
        Shrinkage intensity (0 = no shrinkage, 1 = full shrinkage to target)
    annualization : int
        Annualization factor
        
    Returns:
    --------
    pd.DataFrame : Combined EWMA with shrinkage covariance matrix
    """
    assets = returns.columns
    
    # Calculate EWMA covariance (without annualization)
    ewma_cov = ewma_covariance(returns, lambda_=lambda_, annualization=1)
    
    # Create shrinkage target (constant correlation matrix)
    # Get the average correlation from the EWMA correlation matrix
    corr = ewma_correlation(returns, lambda_=lambda_)
    avg_corr = corr.values.sum() / (corr.shape[0]**2 - corr.shape[0])
    
    # Create the target correlation matrix
    target_corr = np.full_like(corr, avg_corr)
    np.fill_diagonal(target_corr, 1.0)
    
    # Convert correlation target to covariance target using EWMA variances
    variances = np.diag(ewma_cov)
    target_cov = target_corr * np.sqrt(np.outer(variances, variances))
    
    # Apply shrinkage
    shrunk_cov = (1 - shrinkage_intensity) * ewma_cov.values + shrinkage_intensity * target_cov
    
    # Convert to DataFrame
    combined_cov = pd.DataFrame(shrunk_cov, index=assets, columns=assets) * annualization
    
    return combined_cov
