"""Eigenstructure and PCA analysis."""
import numpy as np

def compute_covariance(data):
    """Compute centered data and covariance matrix."""
    mean_vec = np.mean(data, axis=0)
    centered = data - mean_vec
    cov_matrix = np.cov(centered, rowvar=False)
    return mean_vec, centered, cov_matrix

def eigendecompose(matrix):
    """Eigendecomposition sorted by descending eigenvalue."""
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    idx = eigenvalues.argsort()[::-1]
    return eigenvalues[idx].real, eigenvectors[:, idx].real

def analyze_eigenstructure(data):
    """Complete eigenstructure analysis."""
    mean_vec, centered, cov_matrix = compute_covariance(data)
    eigenvalues, eigenvectors = eigendecompose(cov_matrix)
    explained_var = eigenvalues / eigenvalues.sum()
    return {
        'mean': mean_vec, 'centered': centered, 'covariance': cov_matrix,
        'eigenvalues': eigenvalues, 'eigenvectors': eigenvectors, 'explained_variance': explained_var
    }

def pca_transform(data, n_components=2):
    """Apply PCA dimensionality reduction."""
    result = analyze_eigenstructure(data)
    W = result['eigenvectors'][:, :n_components]
    projected = result['centered'] @ W
    return projected, result['explained_variance'][:n_components], W

def reconstruct_data(projected, components, mean_vec):
    """Reconstruct data from PCA projection."""
    return (projected @ components.T) + mean_vec

def compute_reconstruction_error(original, reconstructed):
    """Compute reconstruction error."""
    diff = original - reconstructed
    mse = np.mean(diff ** 2)
    return {'mse': mse, 'rmse': np.sqrt(mse), 'relative_percent': mse / np.var(original) * 100}
