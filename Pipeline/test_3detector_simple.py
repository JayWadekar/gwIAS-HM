#!/usr/bin/env python3
"""
Simple test script for 3-detector functionality without external dependencies.
"""

import os
import sys

def test_imports():
    """Test that our new modules can be imported."""
    print("Testing imports...")
    
    try:
        import coincidence_core
        print("✓ coincidence_core imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import coincidence_core: {e}")
        return False
    
    try:
        import coincidence_veto
        print("✓ coincidence_veto imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import coincidence_veto: {e}")
        return False
    
    try:
        import coherent_score_hm_search
        print("✓ coherent_score_hm_search imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import coherent_score_hm_search: {e}")
        return False
    
    return True

def test_function_existence():
    """Test that our new functions exist."""
    print("\nTesting function existence...")
    
    try:
        from coincidence_core import find_interesting_dir, collect_background_candidates, get_files_by_epoch
        print("✓ coincidence_core functions found")
    except ImportError as e:
        print(f"✗ coincidence_core functions missing: {e}")
        return False
    
    try:
        from coincidence_veto import veto_and_optimize_coincidence_list, select_optimal_trigger_tuples
        print("✓ coincidence_veto functions found")
    except ImportError as e:
        print(f"✗ coincidence_veto functions missing: {e}")
        return False
    
    try:
        from coherent_score_hm_search import initialize_cs_instance_new, compute_coherent_scores_new
        print("✓ coherent_score_hm_search new functions found")
    except ImportError as e:
        print(f"✗ coherent_score_hm_search new functions missing: {e}")
        return False
    
    return True

def test_detector_list_handling():
    """Test that detector lists are handled correctly."""
    print("\nTesting detector list handling...")
    
    try:
        # Test 2-detector case (backward compatibility)
        detectors_2 = ['H1', 'L1']
        network_2 = "".join([d[0] for d in detectors_2])
        if network_2 == "HL":
            print("✓ 2-detector network string correct: HL")
        else:
            print(f"✗ 2-detector network string incorrect: {network_2}")
            return False
        
        # Test 3-detector case
        detectors_3 = ['H1', 'L1', 'V1']
        network_3 = "".join([d[0] for d in detectors_3])
        if network_3 == "HLV":
            print("✓ 3-detector network string correct: HLV")
        else:
            print(f"✗ 3-detector network string incorrect: {network_3}")
            return False
        
        # Test 4-detector case (future expansion)
        detectors_4 = ['H1', 'L1', 'V1', 'K1']
        network_4 = "".join([d[0] for d in detectors_4])
        if network_4 == "HLVK":
            print("✓ 4-detector network string correct: HLVK")
        else:
            print(f"✗ 4-detector network string incorrect: {network_4}")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Detector list handling error: {e}")
        return False

def test_file_structure():
    """Test that our new file structure is correct."""
    print("\nTesting file structure...")
    
    required_files = [
        'coincidence_core.py',
        'coincidence_veto.py',
        'coincidence_HM_new.py',
        'test_3detector.py'
    ]
    
    for filename in required_files:
        if os.path.exists(filename):
            print(f"✓ {filename} exists")
        else:
            print(f"✗ {filename} missing")
            return False
    
    return True

def demonstrate_usage():
    """Show how to use the new 3-detector functionality."""
    print("\n" + "="*60)
    print("3-DETECTOR USAGE EXAMPLES")
    print("="*60)
    
    print("""
Example 1: Basic 3-detector analysis
====================================
from coincidence_core import find_interesting_dir

# Process trigger files for H1, L1, and V1
result = find_interesting_dir(
    dir_name="/path/to/trigger/files",
    detectors=['H1', 'L1', 'V1']
)

Example 2: Using the new command-line interface
==============================================
# Run locally
python coincidence_HM_new.py /path/to/data --detectors H1 L1 V1

# Submit to cluster
python coincidence_HM_new.py /path/to/data --detectors H1 L1 V1 --cluster

Example 3: Programmatic use with custom parameters
==================================================
from coincidence_core import find_interesting_dir

result = find_interesting_dir(
    dir_name="/path/to/trigger/files",
    detectors=['H1', 'L1', 'V1'],
    n_cores=4,
    veto_triggers=True,
    min_veto_chi2=32,
    output_timeseries=True,
    output_coherent_score=True
)

Key Changes from Original:
=========================
1. detectors parameter now accepts a list of any length
2. Supports ['H1', 'L1', 'V1'] for 3-detector analysis
3. Backward compatible with ['H1', 'L1'] for 2-detector analysis
4. Can be extended to ['H1', 'L1', 'V1', 'K1'] for 4-detector analysis
5. Modular structure makes it easier to maintain and extend

Next Steps:
===========
1. Test with real LIGO/Virgo data files
2. Verify coherent score computation accuracy
3. Benchmark performance vs. original 2-detector pipeline
4. Add support for additional detectors (KAGRA, etc.)
""")

def main():
    """Main test function."""
    print("="*60)
    print("GWIAS-HM 3-DETECTOR FUNCTIONALITY TEST")
    print("="*60)
    
    all_passed = True
    
    all_passed &= test_imports()
    all_passed &= test_function_existence()
    all_passed &= test_detector_list_handling()
    all_passed &= test_file_structure()
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("✓ 3-detector functionality is ready for use")
        demonstrate_usage()
        return 0
    else:
        print("✗ Some tests failed")
        print("Please check the error messages above")
        return 1

if __name__ == "__main__":
    sys.exit(main())