"""
Main Entry Point
==================
Script utama untuk menjalankan analisis drone motion.

Author: Ariel Cornelius Sitorus (13524085)
Course: IF2123 Aljabar Linier dan Geometri

Usage:
    python main.py
"""

import numpy as np

# Import modules
from data_generation import generate_linear_flight, generate_spiral_flight
from eigenanalysis import analyze_eigenstructure, pca_transform
from interpretation import analyze_flight_direction, compare_with_ground_truth, print_analysis_report
from visualization import visualize_eigenanalysis, visualize_comparison


def run_linear_flight_analysis():
    """Analyze linear flight trajectory."""
    print("\n" + "=" * 60)
    print("SCENARIO A: LINEAR FLIGHT ANALYSIS")
    print("=" * 60)
    
    # Generate data
    print("\n[1] Generating linear flight data...")
    data = generate_linear_flight(n_samples=500, seed=42)
    print(f"    Data shape: {data.shape}")
    
    # Eigenstructure analysis
    print("\n[2] Performing eigenstructure analysis...")
    analysis = analyze_eigenstructure(data)
    
    print("\n    Covariance Matrix:")
    print(analysis['covariance'])
    
    print("\n    Eigenvalues:")
    for i, ev in enumerate(analysis['eigenvalues']):
        print(f"    L{i+1} = {ev:.4f}")
    
    print("\n    Eigenvectors (columns):")
    print(analysis['eigenvectors'])
    
    # Flight direction interpretation
    print("\n[3] Interpreting flight direction...")
    flight_analysis = analyze_flight_direction(
        analysis['eigenvectors'],
        analysis['eigenvalues']
    )
    
    # Compare with ground truth (45° yaw, 0° pitch from data generation)
    comparison = compare_with_ground_truth(
        estimated_heading=flight_analysis['heading_deg'],
        true_heading=45.0,
        estimated_pitch=flight_analysis['pitch_deg'],
        true_pitch=0.0
    )
    
    print_analysis_report(flight_analysis, comparison)
    
    # Visualization
    print("\n[4] Generating visualization...")
    visualize_eigenanalysis(
        data, analysis,
        title="Linear Flight: Eigenvector Analysis",
        save_path="linear_flight_eigen.png",
        show=False
    )
    
    return data, analysis, flight_analysis


def run_spiral_flight_analysis():
    """Analyze spiral flight trajectory."""
    print("\n" + "=" * 60)
    print("SCENARIO B: SPIRAL FLIGHT ANALYSIS")
    print("=" * 60)
    
    # Generate data
    print("\n[1] Generating spiral flight data...")
    data = generate_spiral_flight(n_samples=500, seed=42)
    print(f"    Data shape: {data.shape}")
    
    # Eigenstructure analysis
    print("\n[2] Performing eigenstructure analysis...")
    analysis = analyze_eigenstructure(data)
    
    print("\n    Eigenvalues:")
    for i, ev in enumerate(analysis['eigenvalues']):
        print(f"    L{i+1} = {ev:.4f}")
    
    # Flight direction interpretation
    print("\n[3] Interpreting flight direction...")
    flight_analysis = analyze_flight_direction(
        analysis['eigenvectors'],
        analysis['eigenvalues']
    )
    
    print_analysis_report(flight_analysis)
    
    # Visualization
    print("\n[4] Generating visualization...")
    visualize_eigenanalysis(
        data, analysis,
        title="Spiral Flight: Eigenvector Analysis",
        save_path="spiral_flight_eigen.png",
        show=False
    )
    
    return data, analysis, flight_analysis


def run_pca_demo():
    """Demonstrate PCA dimensionality reduction."""
    print("\n" + "=" * 60)
    print("PCA DIMENSIONALITY REDUCTION DEMO")
    print("=" * 60)
    
    # Generate data
    data = generate_linear_flight(n_samples=500, seed=42)
    
    # Apply PCA
    print("\n[1] Applying PCA (3D -> 2D)...")
    projected, explained_var, components = pca_transform(data, n_components=2)
    
    print(f"\n    Original shape: {data.shape}")
    print(f"    Projected shape: {projected.shape}")
    print(f"\n    Explained variance:")
    print(f"    PC1: {explained_var[0]*100:.2f}%")
    print(f"    PC2: {explained_var[1]*100:.2f}%")
    print(f"    Total: {explained_var.sum()*100:.2f}%")
    
    return projected, explained_var


def run_comparison():
    """Compare linear vs spiral flight."""
    print("\n" + "=" * 60)
    print("COMPARISON: LINEAR vs SPIRAL FLIGHT")
    print("=" * 60)
    
    # Generate both datasets
    linear_data = generate_linear_flight(n_samples=500)
    spiral_data = generate_spiral_flight(n_samples=500)
    
    # Analyze both
    linear_analysis = analyze_eigenstructure(linear_data)
    spiral_analysis = analyze_eigenstructure(spiral_data)
    
    # Compare eigenvalue ratios
    linear_ratio = linear_analysis['eigenvalues'][0] / linear_analysis['eigenvalues'][1]
    spiral_ratio = spiral_analysis['eigenvalues'][0] / spiral_analysis['eigenvalues'][1]
    
    print(f"\n    Linear flight L1/L2 ratio: {linear_ratio:.2f}")
    print(f"    Spiral flight L1/L2 ratio: {spiral_ratio:.2f}")
    print(f"\n    Linear: Dominant single direction")
    print(f"    Spiral: Near-degenerate (circular symmetry)")
    
    # Side-by-side visualization
    visualize_comparison(
        [(linear_data, linear_analysis), (spiral_data, spiral_analysis)],
        ['Linear Flight', 'Spiral Flight'],
        main_title='Comparison of Flight Trajectories',
        save_path='flight_comparison.png',
        show=False
    )


def main():
    """Main entry point."""
    print("=" * 60)
    print("DRONE MOTION ANALYSIS USING LINEAR ALGEBRA")
    print("Eigenvectors as Directions of Meaning")
    print("=" * 60)
    
    # Run analyses
    run_linear_flight_analysis()
    run_spiral_flight_analysis()
    run_pca_demo()
    run_comparison()
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
