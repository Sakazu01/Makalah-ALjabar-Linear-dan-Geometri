"""
Data Generation Module
=======================
Modul untuk membangkitkan data sintetis posisi drone.

Author: Ariel Cornelius Sitorus (13524085)
Course: IF2123 Aljabar Linier dan Geometri
"""

import numpy as np


def generate_linear_flight(n_samples=500, seed=42):
    """
    Generate synthetic linear flight drone data.
    
    Creates anisotropic data simulating drone moving
    in a dominant direction with some drift and noise.
    
    Parameters
    ----------
    n_samples : int
        Number of position samples
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    np.ndarray
        Drone positions with shape (n_samples, 3)
    """
    np.random.seed(seed)
    
    # White noise (unit sphere)
    white_data = np.random.randn(3, n_samples)
    
    # Scaling matrix - defines variance in each axis
    # sigma_1=10 (primary), sigma_2=3 (drift), sigma_3=1 (noise)
    scale = np.array([
        [10, 0, 0],
        [0,  3, 0],
        [0,  0, 1]
    ])
    
    # Rotation matrices
    theta = np.radians(45)
    c, s = np.cos(theta), np.sin(theta)
    
    rotation_z = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])
    
    # Composite transformation: R_z @ S (only Z rotation for 45 deg heading)
    transform = rotation_z @ scale
    drone_data = transform @ white_data
    
    return drone_data.T


def generate_spiral_flight(n_samples=500, seed=42, 
                           radius=10, height=30, n_revolutions=3):
    """
    Generate spiral ascent drone trajectory.
    
    Parameters
    ----------
    n_samples : int
        Number of position samples
    seed : int
        Random seed for reproducibility
    radius : float
        Spiral radius
    height : float
        Total height of spiral
    n_revolutions : int
        Number of spiral revolutions
        
    Returns
    -------
    np.ndarray
        Drone positions with shape (n_samples, 3)
    """
    np.random.seed(seed)
    
    t = np.linspace(0, n_revolutions * 2 * np.pi, n_samples)
    
    x = radius * np.cos(t) + np.random.randn(n_samples) * 0.5
    y = radius * np.sin(t) + np.random.randn(n_samples) * 0.5
    z = (height / (n_revolutions * 2 * np.pi)) * t + np.random.randn(n_samples) * 0.3
    
    return np.column_stack([x, y, z])


def generate_custom_flight(n_samples=500, seed=42,
                           scale_factors=(10, 3, 1),
                           yaw_deg=45, pitch_deg=0):
    """
    Generate custom anisotropic flight data.
    
    Parameters
    ----------
    n_samples : int
        Number of samples
    seed : int
        Random seed
    scale_factors : tuple
        Variance factors (primary, drift, noise)
    yaw_deg : float
        Yaw angle in degrees
    pitch_deg : float
        Pitch angle in degrees
        
    Returns
    -------
    np.ndarray
        Drone positions with shape (n_samples, 3)
    """
    np.random.seed(seed)
    
    white_data = np.random.randn(3, n_samples)
    
    s1, s2, s3 = scale_factors
    scale = np.array([
        [s1, 0,  0],
        [0,  s2, 0],
        [0,  0,  s3]
    ])
    
    # Yaw rotation (about Z)
    yaw = np.radians(yaw_deg)
    cy, sy = np.cos(yaw), np.sin(yaw)
    R_z = np.array([
        [cy, -sy, 0],
        [sy,  cy, 0],
        [0,   0,  1]
    ])
    
    # Pitch rotation (about Y)
    pitch = np.radians(pitch_deg)
    cp, sp = np.cos(pitch), np.sin(pitch)
    R_y = np.array([
        [ cp, 0, sp],
        [ 0,  1, 0],
        [-sp, 0, cp]
    ])
    
    transform = R_y @ R_z @ scale
    drone_data = transform @ white_data
    
    return drone_data.T
