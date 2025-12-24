# Eigenvectors as Directions of Meaning in Drone Motion Analysis

> IF2123 Linear Algebra and Geometry - Final Paper  
> Institut Teknologi Bandung

## Author
**Ariel Cornelius Sitorus** - 13524085

## Overview
This project demonstrates the geometric interpretation of eigenvectors and eigenvalues applied to drone flight trajectory analysis. The dominant eigenvector of the covariance matrix indicates the primary flight direction, while eigenvalues represent variance magnitude in each direction.

## Project Structure
```
├── docs/                           # LaTeX paper and figures
│   ├── 13524085_*.tex             # Main paper (IEEE format)
│   ├── linear_flight_eigen.png    # Linear trajectory visualization
│   ├── spiral_flight_eigen.png    # Spiral trajectory visualization
│   ├── flight_comparison.png      # Side-by-side comparison
│   └── ttd.jpeg                   # Signature
│
├── src/                           # Python implementation
│   ├── data_generation.py         # Synthetic drone data generation
│   ├── eigenanalysis.py           # Eigenstructure & PCA analysis
│   ├── interpretation.py          # Flight dynamics interpretation
│   ├── visualization.py           # 3D visualization functions
│   ├── main.py                    # Main entry point
│   └── drone_simulation.py        # Legacy (backward compatible)
│
└── README.md
```

## Requirements
- Python 3.8+
- NumPy
- Matplotlib

## Usage
```bash
cd src
python main.py
```

## Key Results
- Heading angle estimation: **43.58°** (error: 1.42° from 45° ground truth)
- Primary eigenvalue captures **91%** of total variance
- Successfully distinguishes linear vs spiral trajectories

## References
- [Rinaldi Munir - IF2123 Course Materials](https://informatika.stei.itb.ac.id/~rinaldi.munir/AlsGeo/alsGeo.htm)
- [3Blue1Brown - Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
