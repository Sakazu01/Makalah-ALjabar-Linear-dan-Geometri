"""
Drone Simulation Module (Legacy)
=================================
File ini sudah di-refactor menjadi modul-modul terpisah.

Struktur baru:
- data_generation.py : Generate synthetic drone data
- eigenanalysis.py   : Eigenstructure & PCA analysis
- interpretation.py  : Drone dynamics interpretation
- visualization.py   : 3D visualization
- main.py            : Main entry point

Untuk menjalankan analisis, gunakan:
    python main.py

Author: Ariel Cornelius Sitorus (13524085)
Course: IF2123 Aljabar Linier dan Geometri
"""

# Re-export untuk backward compatibility
from data_generation import generate_linear_flight, generate_spiral_flight
from eigenanalysis import analyze_eigenstructure, pca_transform
from interpretation import analyze_flight_direction
from visualization import visualize_eigenanalysis


def generate_drone_data(n_samples=500, seed=42):
    """
    Legacy function - redirects to generate_linear_flight.
    
    For new code, use:
        from data_generation import generate_linear_flight
    """
    return generate_linear_flight(n_samples, seed)


if __name__ == "__main__":
    print("This file has been refactored.")
    print("Please run 'python main.py' instead.")
    print("\nRunning main.py...")
    from main import main
    main()