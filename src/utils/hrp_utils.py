"""
HRP (Hierarchical Risk Parity) Utility Functions

This module contains the core mathematical functions used in the HRP algorithm,
including correlation distance calculation, quasi-diagonalization, cluster extraction,
and weight calculation functions.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


def clean_correlation_matrix(corr_matrix):
    """
    Clean and validate correlation matrix to ensure it's suitable for HRP calculation.
    
    Parameters
    ----------
    corr_matrix : pd.DataFrame or np.array
        Raw correlation matrix
        
    Returns
    -------
    pd.DataFrame
        Cleaned correlation matrix
    """
    # Convert to pandas DataFrame if needed
    if isinstance(corr_matrix, np.ndarray):
        corr_matrix = pd.DataFrame(corr_matrix)
    
    # Make a copy to avoid modifying the original
    corr_clean = corr_matrix.copy()
    
    # Replace NaN and infinite values with 0
    corr_clean = corr_clean.fillna(0.0)
    corr_clean = corr_clean.replace([np.inf, -np.inf], 0.0)
    
    # Ensure the matrix is symmetric
    corr_clean = (corr_clean + corr_clean.T) / 2
    
    # Ensure diagonal is 1
    np.fill_diagonal(corr_clean.values, 1.0)
    
    # Clip values to valid correlation range
    corr_clean = corr_clean.clip(-1.0, 1.0)
    
    # Check if matrix is positive semi-definite, if not, make it so
    try:
        eigenvals = np.linalg.eigvals(corr_clean.values)
        if np.any(eigenvals < -1e-8):  # Small negative eigenvalues due to numerical errors
            print("Warning: Correlation matrix is not positive semi-definite, applying regularization")
            # Add small positive value to diagonal to make it positive definite
            regularization = max(0.001, -np.min(eigenvals) + 1e-6)
            np.fill_diagonal(corr_clean.values, corr_clean.values.diagonal() + regularization)
            # Renormalize diagonal to 1
            np.fill_diagonal(corr_clean.values, 1.0)
    except np.linalg.LinAlgError:
        print("Warning: Could not check matrix eigenvalues, proceeding with basic cleaning")
    
    return corr_clean


def validate_returns_for_hrp(returns_window):
    """
    Validate and clean returns data for HRP calculation.
    
    Parameters
    ----------
    returns_window : pd.DataFrame
        Returns data for current window
        
    Returns
    -------
    pd.DataFrame
        Cleaned returns data
    """
    # Remove assets with all NaN values
    returns_clean = returns_window.dropna(axis=1, how='all')
    
    # Remove assets with zero variance (constant values)
    asset_stds = returns_clean.std()
    valid_assets = asset_stds[asset_stds > 1e-8].index
    returns_clean = returns_clean[valid_assets]
    
    # Fill remaining NaN values with 0
    returns_clean = returns_clean.fillna(0.0)
    
    # Remove extreme outliers (more than 5 standard deviations)
    for col in returns_clean.columns:
        col_data = returns_clean[col]
        mean_val = col_data.mean()
        std_val = col_data.std()
        if std_val > 0:
            lower_bound = mean_val - 5 * std_val
            upper_bound = mean_val + 5 * std_val
            returns_clean[col] = col_data.clip(lower_bound, upper_bound)
    
    return returns_clean


def corr_distance(corr):
    """
    Calculate correlation distance matrix with robust handling of non-finite values.
    
    Parameters
    ----------
    corr : np.array or pd.DataFrame
        Correlation matrix
        
    Returns
    -------
    np.array
        Distance matrix based on correlation
    """
    # Ensure we have a numpy array
    if hasattr(corr, 'values'):
        corr = corr.values
    
    # Replace any non-finite values with 0 (uncorrelated)
    corr = np.where(np.isfinite(corr), corr, 0.0)
    
    # Ensure diagonal is 1 (perfect self-correlation)
    np.fill_diagonal(corr, 1.0)
    
    # Clip correlation values to valid range [-1, 1]
    corr = np.clip(corr, -1.0, 1.0)
    
    # Calculate distance: sqrt((1 - corr) / 2)
    distance = np.sqrt((1 - corr) / 2)
    
    # Ensure distance matrix is valid
    distance = np.where(np.isfinite(distance), distance, 1.0)  # Max distance for invalid correlations
    
    return distance


def get_quasi_diag(link):
    """
    Get quasi-diagonal ordering from hierarchical clustering linkage.
    
    Parameters
    ----------
    link : np.array
        Linkage matrix from scipy hierarchical clustering
        
    Returns
    -------
    list
        Quasi-diagonal ordering of assets
    """
    link = link.astype(int)
    n = link.shape[0] + 1  # Number of observations
    
    def recurse(i):
        if i < n:
            return [i]
        j = i - n
        if j >= link.shape[0]:  # Safety check to prevent index out of bounds
            return []
        left, right = link[j, :2]
        return recurse(left) + recurse(right)
    
    return recurse(2*n - 2)


def extract_clusters(link, labels, n_clusters=8):
    """
    Extract n_clusters from hierarchical clustering linkage.
    
    Parameters
    ----------
    link : np.array
        Linkage matrix from scipy
    labels : list
        Asset labels (symbol names)
    n_clusters : int
        Target number of clusters
        
    Returns
    -------
    dict
        Dictionary of clusters with asset indices
    """
    from scipy.cluster.hierarchy import fcluster
    
    # Get cluster assignments
    cluster_assignments = fcluster(link, n_clusters, criterion='maxclust')
    
    # Group assets by cluster
    clusters = {}
    for i, cluster_id in enumerate(cluster_assignments):
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(i)
    
    # Convert to labeled clusters
    labeled_clusters = {}
    for cluster_id, asset_indices in clusters.items():
        cluster_assets = []
        for idx in asset_indices:
            if idx < len(labels):
                cluster_assets.append(labels[idx])
        labeled_clusters[f'Cluster_{cluster_id}'] = cluster_assets
    
    return labeled_clusters


def get_cluster_var(cov, cluster_items):
    """
    Calculate cluster variance.
    
    Parameters
    ----------
    cov : pd.DataFrame
        Covariance matrix
    cluster_items : list
        List of assets in the cluster
        
    Returns
    -------
    float
        Cluster variance
    """
    # Get the covariance sub-matrix for this cluster
    cluster_cov = cov.loc[cluster_items, cluster_items]
    
    # Calculate inverse-variance weights
    inv_diag = 1 / np.diag(cluster_cov)
    inv_diag /= inv_diag.sum()
    
    # Calculate cluster variance
    cluster_var = np.dot(inv_diag, np.dot(cluster_cov, inv_diag))
    
    return cluster_var


def select_best_assets(clusters, returns_data, n_per_cluster=2):
    """
    Select the n best assets from each cluster based on Sharpe ratio.
    
    Parameters
    ----------
    clusters : dict
        Dictionary with cluster IDs as keys and lists of assets as values
    returns_data : pd.DataFrame
        Returns data for calculating metrics
    n_per_cluster : int
        Number of assets to select from each cluster
        
    Returns
    -------
    list
        List of selected assets
    """
    selected_assets = []
    
    # Calculate Sharpe ratios for all assets at once (more efficient)
    mean_returns = returns_data.mean() * 252  # Annualized return
    std_returns = returns_data.std() * np.sqrt(252)  # Annualized volatility
    # Handle case where std is 0
    std_returns = std_returns.replace(0, 1e-8)
    all_sharpe_ratios = mean_returns / std_returns
    
    # First, select the top assets from each cluster
    for cluster_name, assets in clusters.items():
        # Skip empty clusters
        if len(assets) == 0:
            continue

        # Filter to assets in this cluster
        available_assets = [a for a in assets if a in all_sharpe_ratios.index]
        
        if not available_assets:
            continue
            
        cluster_sharpes = all_sharpe_ratios.loc[available_assets]

        # Select top n_per_cluster assets by Sharpe ratio
        n_to_select = min(n_per_cluster, len(available_assets))
        top_assets = cluster_sharpes.nlargest(n_to_select).index.tolist()
        selected_assets.extend(top_assets)
    
    return selected_assets


def get_hrp_weights(cov, corr, returns_data=None, n_clusters=8, n_per_cluster=2, min_allocation=0.02):
    """
    Calculate HRP weights with asset selection and minimum allocation.
    
    Parameters
    ----------
    cov : pd.DataFrame
        Covariance matrix
    corr : pd.DataFrame
        Correlation matrix
    returns_data : pd.DataFrame
        Returns data for calculating metrics
    n_clusters : int
        Target number of clusters
    n_per_cluster : int
        Number of assets to select from each cluster
    min_allocation : float
        Minimum allocation per position (e.g., 0.02 for 2%)
        
    Returns
    -------
    pd.Series
        Asset weights
    """
    # Clean and validate correlation matrix
    corr = clean_correlation_matrix(corr)
    
    # Calculate distance matrix and perform hierarchical clustering
    dist = corr_distance(corr.values)
    dist = pd.DataFrame(dist, index=corr.index, columns=corr.index)
    
    # Convert to condensed distance matrix for linkage
    # Use average linkage as specified in the user's approach
    condensed_dist = squareform(dist.values, checks=False)
    link = linkage(condensed_dist, method='average')
    
    # Extract clusters
    clusters = extract_clusters(link, list(corr.index), n_clusters)
    
    # Select best assets from each cluster if returns data is provided
    if returns_data is not None:
        # Validate and clean returns data
        returns_data = validate_returns_for_hrp(returns_data)
        
        selected_assets = select_best_assets(clusters, returns_data, n_per_cluster)
        
        # Filter covariance and correlation matrices to selected assets
        valid_assets = [asset for asset in selected_assets if asset in cov.index and asset in corr.index]
        
        if len(valid_assets) < 2:
            # Fallback to all available assets if selection is too restrictive
            valid_assets = list(cov.index)
        
        cov = cov.loc[valid_assets, valid_assets]
        corr = corr.loc[valid_assets, valid_assets]
        
        # Recalculate distance and linkage for selected assets
        dist = corr_distance(corr.values)
        dist = pd.DataFrame(dist, index=corr.index, columns=corr.index)
        condensed_dist = squareform(dist.values, checks=False)
        link = linkage(condensed_dist, method='average')  # Use 'average' consistently
    
    # Get quasi-diagonal ordering
    sort_ix = get_quasi_diag(link)
    sort_ix = [i for i in sort_ix if i < len(cov.index)]  # Ensure indices are valid
    sorted_items = [cov.index[i] for i in sort_ix]
    
    # Recursive bisection for HRP weights - optimized according to user's approach
    weights = pd.Series(1.0, index=sorted_items)
    
    def recurse_bisect(idx_list):
        if len(idx_list) == 1:  # Base case: cannot bisect further
            return
        # Split into two groups
        split = len(idx_list) // 2
        left = idx_list[:split]
        right = idx_list[split:]
        
        # Extract corresponding submatrices of covariance
        cov_l = cov.loc[left, left].values
        cov_r = cov.loc[right, right].values
        
        # Calculate variance of clusters
        var_l = weights[left] @ cov_l @ weights[left]
        var_r = weights[right] @ cov_r @ weights[right]
        
        # Calculate alpha (weight allocation factor)
        alpha = 1 - var_l / (var_l + var_r) if (var_l + var_r) > 0 else 0.5
        
        # Redistribute weights
        weights[left] *= alpha
        weights[right] *= (1 - alpha)
        
        # Continue recursion
        recurse_bisect(left)
        recurse_bisect(right)
    
    # Start recursion with all sorted items
    recurse_bisect(list(sorted_items))
    
    # The weights object now contains the hierarchical risk parity weights
    hrp_weights = weights.copy()
    
    # Apply minimum allocation constraint
    if min_allocation > 0:
        n_assets = len(hrp_weights)
        total_min_allocation = min_allocation * n_assets
        
        if total_min_allocation < 1.0:
            # Scale down weights that are above minimum
            excess_weight = 1.0 - total_min_allocation
            
            # Set minimum allocations
            adjusted_weights = hrp_weights.copy()
            adjusted_weights[adjusted_weights < min_allocation] = min_allocation
            
            # Scale remaining weights
            above_min = adjusted_weights > min_allocation
            if above_min.sum() > 0:
                current_excess = adjusted_weights[above_min].sum() - (above_min.sum() * min_allocation)
                if current_excess > 0:
                    scale_factor = excess_weight / current_excess
                    adjusted_weights[above_min] = min_allocation + (adjusted_weights[above_min] - min_allocation) * scale_factor
            
            hrp_weights = adjusted_weights
    
    # Normalize weights to sum to 1
    hrp_weights = hrp_weights / hrp_weights.sum()
    
    # Create full weight series including all original assets (with 0 weights for non-selected)
    if returns_data is not None:
        full_weights = pd.Series(0.0, index=returns_data.columns)
        full_weights[hrp_weights.index] = hrp_weights
        return full_weights.sort_index()
    
    return hrp_weights.sort_index()


def filter_valid_symbols_for_window(window_returns):
    """
    Filter out symbols with any missing data in the current window.
    
    Parameters
    ----------
    window_returns : pd.DataFrame
        Returns data for the current window
        
    Returns
    -------
    pd.DataFrame
        Filtered returns data with only valid symbols
    """
    # Use the more comprehensive validation function
    return validate_returns_for_hrp(window_returns)
