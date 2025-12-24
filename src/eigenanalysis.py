"""
Eigenanalysis Module
=====================
Modul untuk analisis eigenstructure dan PCA.

Author: Ariel Cornelius Sitorus (13524085)
Course: IF2123 Aljabar Linier dan Geometri
"""

import numpy as np


def compute_covariance(data):
    """
    Compute centered data and covariance matrix.
    
    Parameters
    ----------
    data : np.ndarray
        Input data with shape (n_samples, n_features)
        
    Returns
    -------
    mean_vec : np.ndarray
        Mean vector (centroid)
    centered : np.ndarray
        Centered data matrix
    cov_matrix : np.ndarray
        Sample covariance matrix
    """
    mean_vec = np.mean(data, axis=0)
    centered = data - mean_vec
    cov_matrix = np.cov(centered, rowvar=False)
    
    return mean_vec, centered, cov_matrix


def eigendecompose(matrix):
    """
    Perform eigendecomposition on a matrix.
    
    Returns eigenvalues and eigenvectors sorted
    by descending eigenvalue magnitude.
    
    Parameters
    ----------
    matrix : np.ndarray
        Square matrix to decompose
        
    Returns
    -------
    eigenvalues : np.ndarray
        Sorted eigenvalues (descending)
    eigenvectors : np.ndarray
        Corresponding eigenvectors (columns)
    """
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    
    # Sort by descending eigenvalue
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx].real
    eigenvectors = eigenvectors[:, idx].real
    
    return eigenvalues, eigenvectors


def analyze_eigenstructure(data):
    """
    Complete eigenstructure analysis of data.
    
    Parameters
    ----------
    data : np.ndarray
        Input data (n_samples, n_features)
        
    Returns
    -------
    dict
        Dictionary containing:
        - mean: centroid vector
        - centered: centered data
        - covariance: covariance matrix
        - eigenvalues: sorted eigenvalues
        - eigenvectors: sorted eigenvectors
        - explained_variance: variance ratios
    """
    mean_vec, centered, cov_matrix = compute_covariance(data)
    eigenvalues, eigenvectors = eigendecompose(cov_matrix)
    
    total_var = eigenvalues.sum()
    explained_var = eigenvalues / total_var
    
    return {
        'mean': mean_vec,
        'centered': centered,
        'covariance': cov_matrix,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'explained_variance': explained_var
    }


def pca_transform(data, n_components=2):
    """
    Apply PCA dimensionality reduction.
    
    Parameters
    ----------
    data : np.ndarray
        Input data (n_samples, n_features)
    n_components : int
        Number of principal components
        
    Returns
    -------
    projected : np.ndarray
        Projected data (n_samples, n_components)
    explained_var : np.ndarray
        Explained variance ratios
    components : np.ndarray
        Principal component vectors
    """
    result = analyze_eigenstructure(data)
    
    # Select top-k components
    W = result['eigenvectors'][:, :n_components]
    projected = result['centered'] @ W
    
    explained_var = result['explained_variance'][:n_components]
    
    return projected, explained_var, W


def reconstruct_data(projected, components, mean_vec):
    """
    Reconstruct data from PCA projection.
    
    Parameters
    ----------
    projected : np.ndarray
        Projected data
    components : np.ndarray
        Principal component vectors
    mean_vec : np.ndarray
        Original mean vector
        
    Returns
    -------
    np.ndarray
        Reconstructed data
    """
    return (projected @ components.T) + mean_vec


def compute_reconstruction_error(original, reconstructed):
    """
    Compute reconstruction error metrics.
    
    Parameters
    ----------
    original : np.ndarray
        Original data
    reconstructed : np.ndarray
        Reconstructed data
        
    Returns
    -------
    dict
        Error metrics (MSE, RMSE, relative error)
    """
    diff = original - reconstructed
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    
    original_var = np.var(original)
    relative_error = mse / original_var if original_var > 0 else 0
    
    return {
        'mse': mse,
        'rmse': rmse,
        'relative_error': relative_error,
        'relative_percent': relative_error * 100
    }
