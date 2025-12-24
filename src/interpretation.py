"""
Interpretation Module
======================
Modul untuk interpretasi hasil eigendecomposition dalam konteks drone dynamics.

Author: Ariel Cornelius Sitorus (13524085)
Course: IF2123 Aljabar Linier dan Geometri
"""

import numpy as np


def compute_heading_pitch(eigenvector):
    """
    Compute heading (yaw) and pitch angles from eigenvector.
    
    Parameters
    ----------
    eigenvector : np.ndarray
        3D eigenvector representing direction
        
    Returns
    -------
    heading_deg : float
        Heading angle in degrees (yaw)
    pitch_deg : float
        Pitch angle in degrees
    """
    v = eigenvector
    
    # Heading (yaw) - angle in XY plane
    heading_rad = np.arctan2(v[1], v[0])
    heading_deg = np.degrees(heading_rad)
    
    # Pitch - angle from horizontal plane
    horizontal_mag = np.sqrt(v[0]**2 + v[1]**2)
    pitch_rad = np.arctan2(v[2], horizontal_mag)
    pitch_deg = np.degrees(pitch_rad)
    
    return heading_deg, pitch_deg


def interpret_eigenvalues(eigenvalues):
    """
    Interpret eigenvalues in drone dynamics context.
    
    Parameters
    ----------
    eigenvalues : np.ndarray
        Array of eigenvalues
        
    Returns
    -------
    dict
        Interpretation results
    """
    total = eigenvalues.sum()
    ratios = eigenvalues / total
    
    interpretations = [
        "Primary motion (forward flight)",
        "Lateral drift (wind/turbulence)",
        "Vertical oscillation (altitude noise)"
    ]
    
    results = []
    for i, (val, ratio) in enumerate(zip(eigenvalues, ratios)):
        label = interpretations[i] if i < len(interpretations) else f"Component {i+1}"
        results.append({
            'component': i + 1,
            'eigenvalue': val,
            'variance_ratio': ratio,
            'variance_percent': ratio * 100,
            'interpretation': label
        })
    
    return {
        'total_variance': total,
        'components': results,
        'dominant_ratio': ratios[0] if len(ratios) > 0 else 0
    }


def analyze_flight_direction(eigenvectors, eigenvalues):
    """
    Complete flight direction analysis.
    
    Parameters
    ----------
    eigenvectors : np.ndarray
        Eigenvector matrix (columns are eigenvectors)
    eigenvalues : np.ndarray
        Corresponding eigenvalues
        
    Returns
    -------
    dict
        Complete flight analysis results
    """
    # Primary direction (largest eigenvalue)
    v1 = eigenvectors[:, 0]
    heading, pitch = compute_heading_pitch(v1)
    
    # Eigenvalue interpretation
    eigen_interp = interpret_eigenvalues(eigenvalues)
    
    # Check for degeneracy (similar eigenvalues)
    if len(eigenvalues) >= 2:
        ratio_1_2 = eigenvalues[0] / eigenvalues[1] if eigenvalues[1] > 0 else np.inf
        is_degenerate = ratio_1_2 < 1.5  # Nearly equal
    else:
        ratio_1_2 = np.inf
        is_degenerate = False
    
    return {
        'heading_deg': heading,
        'pitch_deg': pitch,
        'primary_direction': v1,
        'eigenvalue_analysis': eigen_interp,
        'eigenvalue_ratio_1_2': ratio_1_2,
        'is_degenerate': is_degenerate,
        'degeneracy_warning': "Eigenvalues nearly equal - no unique direction" if is_degenerate else None
    }


def compare_with_ground_truth(estimated_heading, true_heading,
                               estimated_pitch=None, true_pitch=None):
    """
    Compare estimated angles with ground truth.
    
    Parameters
    ----------
    estimated_heading : float
        Estimated heading in degrees
    true_heading : float
        True heading in degrees
    estimated_pitch : float, optional
        Estimated pitch in degrees
    true_pitch : float, optional
        True pitch in degrees
        
    Returns
    -------
    dict
        Comparison results with errors
    """
    heading_error = abs(estimated_heading - true_heading)
    # Handle wraparound
    if heading_error > 180:
        heading_error = 360 - heading_error
    
    result = {
        'heading_estimated': estimated_heading,
        'heading_true': true_heading,
        'heading_error': heading_error
    }
    
    if estimated_pitch is not None and true_pitch is not None:
        pitch_error = abs(estimated_pitch - true_pitch)
        result.update({
            'pitch_estimated': estimated_pitch,
            'pitch_true': true_pitch,
            'pitch_error': pitch_error
        })
    
    return result


def print_analysis_report(analysis_result, comparison=None):
    """
    Print formatted analysis report.
    
    Parameters
    ----------
    analysis_result : dict
        Result from analyze_flight_direction
    comparison : dict, optional
        Result from compare_with_ground_truth
    """
    print("\n" + "=" * 50)
    print("DRONE FLIGHT DIRECTION ANALYSIS")
    print("=" * 50)
    
    print(f"\nEstimated Heading (Yaw): {analysis_result['heading_deg']:.2f} deg")
    print(f"Estimated Pitch: {analysis_result['pitch_deg']:.2f} deg")
    
    print(f"\nPrimary Direction Vector: {analysis_result['primary_direction']}")
    
    if analysis_result['is_degenerate']:
        print(f"\n[!] WARNING: {analysis_result['degeneracy_warning']}")
    
    print("\n--- Variance Analysis ---")
    for comp in analysis_result['eigenvalue_analysis']['components']:
        print(f"L{comp['component']} = {comp['eigenvalue']:.2f} "
              f"({comp['variance_percent']:.1f}%) - {comp['interpretation']}")
    
    if comparison:
        print("\n--- Ground Truth Comparison ---")
        print(f"Heading Error: {comparison['heading_error']:.2f} deg")
        if 'pitch_error' in comparison:
            print(f"Pitch Error: {comparison['pitch_error']:.2f} deg")
    
    print("=" * 50)
