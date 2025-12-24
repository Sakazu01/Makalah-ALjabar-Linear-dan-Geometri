from data_generation import generate_linear_flight, generate_spiral_flight
from eigenanalysis import analyze_eigenstructure, pca_transform
from interpretation import analyze_flight_direction
from visualization import visualize_eigenanalysis

def generate_drone_data(n_samples=500, seed=42):
    return generate_linear_flight(n_samples, seed)

if __name__ == "__main__":
    print("Run 'python main.py' instead.")
    from main import main
    main()