"""Data generation for drone simulation."""
import numpy as np

def generate_linear_flight(n_samples=500, seed=42):
    """Generate anisotropic linear flight data with 45deg heading."""
    np.random.seed(seed)
    white_data = np.random.randn(3, n_samples)
    
    scale = np.array([[10, 0, 0], [0, 3, 0], [0, 0, 1]])
    
    theta = np.radians(45)
    c, s = np.cos(theta), np.sin(theta)
    rotation_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
    transform = rotation_z @ scale
    return (transform @ white_data).T

def generate_spiral_flight(n_samples=500, seed=42, radius=10, height=30, n_revolutions=3):
    """Generate spiral ascent trajectory."""
    np.random.seed(seed)
    t = np.linspace(0, n_revolutions * 2 * np.pi, n_samples)
    
    x = radius * np.cos(t) + np.random.randn(n_samples) * 0.5
    y = radius * np.sin(t) + np.random.randn(n_samples) * 0.5
    z = (height / (n_revolutions * 2 * np.pi)) * t + np.random.randn(n_samples) * 0.3
    
    return np.column_stack([x, y, z])

def generate_custom_flight(n_samples=500, seed=42, scale_factors=(10, 3, 1), yaw_deg=45, pitch_deg=0):
    """Generate custom anisotropic flight data."""
    np.random.seed(seed)
    white_data = np.random.randn(3, n_samples)
    
    s1, s2, s3 = scale_factors
    scale = np.array([[s1, 0, 0], [0, s2, 0], [0, 0, s3]])
    
    yaw, pitch = np.radians(yaw_deg), np.radians(pitch_deg)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    
    R_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    R_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    
    transform = R_y @ R_z @ scale
    return (transform @ white_data).T
