"""3D visualization for drone eigenanalysis."""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_point_cloud(ax, data, alpha=0.3, s=5, color='blue', label='Data'):
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], alpha=alpha, s=s, color=color, label=label)

def plot_eigenvectors(ax, origin, eigenvectors, eigenvalues, scale=2.0, colors=None, labels=None):
    colors = colors or ['red', 'green', 'blue']
    for i in range(min(len(eigenvalues), eigenvectors.shape[1])):
        length = np.sqrt(eigenvalues[i]) * scale
        lbl = labels[i] if labels else f'v{i+1} ({eigenvalues[i]:.1f})'
        ax.quiver(*origin, *(eigenvectors[:, i] * length), color=colors[i % 3], linewidth=3, label=lbl)

def set_equal_aspect(ax, data):
    max_range = np.array([data[:, i].max() - data[:, i].min() for i in range(3)]).max() / 2.0
    mid = [(data[:, i].max() + data[:, i].min()) * 0.5 for i in range(3)]
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

def visualize_eigenanalysis(data, analysis, title=None, save_path=None, show=True):
    """Main visualization function."""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    mean = analysis['mean']
    evals, evecs = analysis['eigenvalues'], analysis['eigenvectors']
    explained = analysis['explained_variance']
    labels = [f'v{i+1}: {explained[i]*100:.1f}%' for i in range(len(evals))]
    
    plot_point_cloud(ax, data, label='Positions')
    ax.scatter(*mean, color='red', s=100, label='Centroid')
    plot_eigenvectors(ax, mean, evecs, evals, labels=labels)
    
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(title or 'Eigenvector Analysis')
    ax.legend(loc='upper left')
    set_equal_aspect(ax, data)
    plt.tight_layout()
    
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show: plt.show()
    return fig

def visualize_comparison(data_list, titles, main_title=None, save_path=None, show=True):
    """Compare multiple datasets."""
    n = len(data_list)
    fig = plt.figure(figsize=(6*n, 6))
    
    for i, (data, analysis) in enumerate(data_list):
        ax = fig.add_subplot(1, n, i+1, projection='3d')
        plot_point_cloud(ax, data)
        ax.scatter(*analysis['mean'], color='red', s=100)
        plot_eigenvectors(ax, analysis['mean'], analysis['eigenvectors'], analysis['eigenvalues'])
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title(titles[i])
        set_equal_aspect(ax, data)
    
    if main_title: fig.suptitle(main_title, fontsize=14)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show: plt.show()
    return fig
