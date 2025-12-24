"""
Visualization Module
=====================
Modul untuk visualisasi 3D data drone dan eigenvector.

Author: Ariel Cornelius Sitorus (13524085)
Course: IF2123 Aljabar Linier dan Geometri
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_point_cloud(ax, data, alpha=0.3, s=5, color='blue', label='Data'):
    """
    Plot 3D point cloud.
    
    Parameters
    ----------
    ax : Axes3D
        Matplotlib 3D axes
    data : np.ndarray
        Data points (n_samples, 3)
    alpha : float
        Point transparency
    s : float
        Point size
    color : str
        Point color
    label : str
        Legend label
    """
    ax.scatter(data[:, 0], data[:, 1], data[:, 2],
               alpha=alpha, s=s, color=color, label=label)


def plot_eigenvectors(ax, origin, eigenvectors, eigenvalues, 
                      scale_factor=2.0, colors=None, labels=None):
    """
    Plot eigenvector arrows from origin.
    
    Parameters
    ----------
    ax : Axes3D
        Matplotlib 3D axes
    origin : np.ndarray
        Origin point (centroid)
    eigenvectors : np.ndarray
        Eigenvector matrix (columns)
    eigenvalues : np.ndarray
        Corresponding eigenvalues
    scale_factor : float
        Arrow length scale factor
    colors : list, optional
        Colors for each eigenvector
    labels : list, optional
        Labels for each eigenvector
    """
    if colors is None:
        colors = ['red', 'green', 'blue']
    
    if labels is None:
        labels = [f'v{i+1} (λ={eigenvalues[i]:.1f})' 
                  for i in range(len(eigenvalues))]
    
    n_vecs = min(len(eigenvalues), eigenvectors.shape[1])
    
    for i in range(n_vecs):
        direction = eigenvectors[:, i]
        length = np.sqrt(eigenvalues[i]) * scale_factor
        
        ax.quiver(origin[0], origin[1], origin[2],
                  direction[0] * length,
                  direction[1] * length,
                  direction[2] * length,
                  color=colors[i % len(colors)],
                  linewidth=3,
                  label=labels[i])


def plot_centroid(ax, centroid, color='red', s=100, label='Centroid'):
    """
    Plot centroid marker.
    
    Parameters
    ----------
    ax : Axes3D
        Matplotlib 3D axes
    centroid : np.ndarray
        Centroid coordinates
    color : str
        Marker color
    s : float
        Marker size
    label : str
        Legend label
    """
    ax.scatter(centroid[0], centroid[1], centroid[2],
               color=color, s=s, marker='o', label=label)


def set_equal_aspect(ax, data):
    """
    Set equal aspect ratio for 3D plot.
    
    Parameters
    ----------
    ax : Axes3D
        Matplotlib 3D axes
    data : np.ndarray
        Data used to determine bounds
    """
    max_range = np.array([
        data[:, 0].max() - data[:, 0].min(),
        data[:, 1].max() - data[:, 1].min(),
        data[:, 2].max() - data[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (data[:, 0].max() + data[:, 0].min()) * 0.5
    mid_y = (data[:, 1].max() + data[:, 1].min()) * 0.5
    mid_z = (data[:, 2].max() + data[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


def visualize_eigenanalysis(data, analysis_result, title=None, 
                            save_path=None, show=True):
    """
    Complete eigenanalysis visualization.
    
    Parameters
    ----------
    data : np.ndarray
        Original data points
    analysis_result : dict
        Result from eigenanalysis.analyze_eigenstructure()
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display plot
        
    Returns
    -------
    fig : Figure
        Matplotlib figure object
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract from analysis result
    mean_vec = analysis_result['mean']
    eigenvalues = analysis_result['eigenvalues']
    eigenvectors = analysis_result['eigenvectors']
    explained_var = analysis_result['explained_variance']
    
    # Create labels with variance info
    labels = [
        f'v{i+1}: {explained_var[i]*100:.1f}% variance'
        for i in range(len(eigenvalues))
    ]
    
    # Plot components
    plot_point_cloud(ax, data, label='Drone Positions')
    plot_centroid(ax, mean_vec)
    plot_eigenvectors(ax, mean_vec, eigenvectors, eigenvalues, labels=labels)
    
    # Styling
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Eigenvector Analysis of Drone Flight Data')
    
    ax.legend(loc='upper left')
    set_equal_aspect(ax, data)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def visualize_pca_projection(original, projected, components, mean_vec,
                             title=None, save_path=None, show=True):
    """
    Visualize PCA projection results.
    
    Parameters
    ----------
    original : np.ndarray
        Original 3D data
    projected : np.ndarray
        Projected 2D data
    components : np.ndarray
        Principal components used
    mean_vec : np.ndarray
        Mean vector
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display
        
    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Original 3D data
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(original[:, 0], original[:, 1], original[:, 2],
                alpha=0.3, s=5)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Original 3D Data')
    
    # Projected 2D data
    axes[1].scatter(projected[:, 0], projected[:, 1], alpha=0.5, s=10)
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    axes[1].set_title('PCA Projection (2D)')
    axes[1].axis('equal')
    axes[1].grid(True, alpha=0.3)
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def visualize_comparison(data_list, titles, main_title=None,
                         save_path=None, show=True):
    """
    Compare multiple datasets side by side.
    
    Parameters
    ----------
    data_list : list
        List of (data, analysis_result) tuples
    titles : list
        Titles for each subplot
    main_title : str, optional
        Main figure title
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display
        
    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    n = len(data_list)
    fig = plt.figure(figsize=(6*n, 6))
    
    for i, (data, analysis) in enumerate(data_list):
        ax = fig.add_subplot(1, n, i+1, projection='3d')
        
        mean_vec = analysis['mean']
        eigenvalues = analysis['eigenvalues']
        eigenvectors = analysis['eigenvectors']
        
        plot_point_cloud(ax, data)
        plot_centroid(ax, mean_vec)
        plot_eigenvectors(ax, mean_vec, eigenvectors, eigenvalues)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(titles[i])
        set_equal_aspect(ax, data)
    
    if main_title:
        fig.suptitle(main_title, fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig
