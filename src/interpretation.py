"""Flight dynamics interpretation from eigendecomposition."""
import numpy as np

def compute_heading_pitch(eigenvector):
    """Compute heading (yaw) and pitch angles from eigenvector."""
    v = eigenvector
    heading_deg = np.degrees(np.arctan2(v[1], v[0]))
    pitch_deg = np.degrees(np.arctan2(v[2], np.sqrt(v[0]**2 + v[1]**2)))
    return heading_deg, pitch_deg

def interpret_eigenvalues(eigenvalues):
    """Interpret eigenvalues in drone dynamics context."""
    total = eigenvalues.sum()
    ratios = eigenvalues / total
    labels = ["Primary motion", "Lateral drift", "Vertical noise"]
    
    return [{
        'component': i+1, 'eigenvalue': val, 'variance_percent': ratio * 100,
        'interpretation': labels[i] if i < 3 else f"PC{i+1}"
    } for i, (val, ratio) in enumerate(zip(eigenvalues, ratios))]

def analyze_flight_direction(eigenvectors, eigenvalues):
    """Complete flight direction analysis."""
    heading, pitch = compute_heading_pitch(eigenvectors[:, 0])
    ratio = eigenvalues[0] / eigenvalues[1] if eigenvalues[1] > 0 else np.inf
    
    return {
        'heading_deg': heading, 'pitch_deg': pitch,
        'primary_direction': eigenvectors[:, 0],
        'eigenvalue_analysis': interpret_eigenvalues(eigenvalues),
        'eigenvalue_ratio_1_2': ratio, 'is_degenerate': ratio < 1.5
    }

def compare_with_ground_truth(estimated_heading, true_heading, estimated_pitch=None, true_pitch=None):
    """Compare estimated angles with ground truth."""
    heading_error = abs(estimated_heading - true_heading)
    if heading_error > 180: heading_error = 360 - heading_error
    
    result = {'heading_estimated': estimated_heading, 'heading_true': true_heading, 'heading_error': heading_error}
    if estimated_pitch is not None and true_pitch is not None:
        result.update({'pitch_estimated': estimated_pitch, 'pitch_true': true_pitch, 'pitch_error': abs(estimated_pitch - true_pitch)})
    return result

def print_analysis_report(analysis_result, comparison=None):
    """Print formatted analysis report."""
    print("\n" + "=" * 50)
    print("DRONE FLIGHT DIRECTION ANALYSIS")
    print("=" * 50)
    print(f"\nHeading: {analysis_result['heading_deg']:.2f} deg")
    print(f"Pitch: {analysis_result['pitch_deg']:.2f} deg")
    print(f"Direction: {analysis_result['primary_direction']}")
    
    if analysis_result['is_degenerate']:
        print("\n[!] WARNING: Near-degenerate eigenvalues")
    
    print("\n--- Variance ---")
    for c in analysis_result['eigenvalue_analysis']:
        print(f"L{c['component']}: {c['eigenvalue']:.2f} ({c['variance_percent']:.1f}%) - {c['interpretation']}")
    
    if comparison:
        print(f"\n--- Error ---\nHeading: {comparison['heading_error']:.2f} deg")
        if 'pitch_error' in comparison: print(f"Pitch: {comparison['pitch_error']:.2f} deg")
    print("=" * 50)
