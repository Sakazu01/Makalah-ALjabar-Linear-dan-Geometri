"""Main script for drone motion eigenanalysis."""
from data_generation import generate_linear_flight, generate_spiral_flight
from eigenanalysis import analyze_eigenstructure, pca_transform
from interpretation import analyze_flight_direction, compare_with_ground_truth, print_analysis_report
from visualization import visualize_eigenanalysis, visualize_comparison

def run_linear_analysis():
    print("\n" + "="*60 + "\nLINEAR FLIGHT ANALYSIS\n" + "="*60)
    data = generate_linear_flight(n_samples=500, seed=42)
    analysis = analyze_eigenstructure(data)
    
    print(f"\nCovariance:\n{analysis['covariance']}")
    print(f"\nEigenvalues: {analysis['eigenvalues']}")
    print(f"Eigenvectors:\n{analysis['eigenvectors']}")
    
    flight = analyze_flight_direction(analysis['eigenvectors'], analysis['eigenvalues'])
    comparison = compare_with_ground_truth(flight['heading_deg'], 45.0, flight['pitch_deg'], 0.0)
    print_analysis_report(flight, comparison)
    
    visualize_eigenanalysis(data, analysis, "Linear Flight", "linear_flight_eigen.png", show=False)
    return data, analysis

def run_spiral_analysis():
    print("\n" + "="*60 + "\nSPIRAL FLIGHT ANALYSIS\n" + "="*60)
    data = generate_spiral_flight(n_samples=500, seed=42)
    analysis = analyze_eigenstructure(data)
    
    print(f"\nEigenvalues: {analysis['eigenvalues']}")
    flight = analyze_flight_direction(analysis['eigenvectors'], analysis['eigenvalues'])
    print_analysis_report(flight)
    
    visualize_eigenanalysis(data, analysis, "Spiral Flight", "spiral_flight_eigen.png", show=False)
    return data, analysis

def run_pca_demo():
    print("\n" + "="*60 + "\nPCA DEMO\n" + "="*60)
    data = generate_linear_flight(n_samples=500, seed=42)
    projected, explained, _ = pca_transform(data, n_components=2)
    print(f"\nOriginal: {data.shape}, Projected: {projected.shape}")
    print(f"Explained variance: PC1={explained[0]*100:.2f}%, PC2={explained[1]*100:.2f}%, Total={explained.sum()*100:.2f}%")

def run_comparison():
    print("\n" + "="*60 + "\nCOMPARISON\n" + "="*60)
    linear = generate_linear_flight(), analyze_eigenstructure(generate_linear_flight())
    spiral = generate_spiral_flight(), analyze_eigenstructure(generate_spiral_flight())
    
    r1 = linear[1]['eigenvalues'][0] / linear[1]['eigenvalues'][1]
    r2 = spiral[1]['eigenvalues'][0] / spiral[1]['eigenvalues'][1]
    print(f"\nLinear L1/L2: {r1:.2f}, Spiral L1/L2: {r2:.2f}")
    
    visualize_comparison([linear, spiral], ['Linear', 'Spiral'], 'Comparison', 'flight_comparison.png', show=False)

def main():
    print("="*60 + "\nDRONE MOTION EIGENANALYSIS\n" + "="*60)
    run_linear_analysis()
    run_spiral_analysis()
    run_pca_demo()
    run_comparison()
    print("\n" + "="*60 + "\nDONE\n" + "="*60)

if __name__ == "__main__":
    main()
