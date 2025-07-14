"""
Mock cogwheel module for testing purposes.
This provides minimal functionality to allow our tests to run without the full cogwheel installation.
"""

import numpy as np

class MockSkyDictionary:
    """Mock SkyDictionary class for testing."""
    
    def __init__(self, network_string):
        self.network_string = network_string
        self.detectors = list(network_string)
        
    def __str__(self):
        return f"MockSkyDictionary({self.network_string})"

class MockLikelihoodMarginalization:
    """Mock likelihood marginalization module."""
    
    @staticmethod
    def SkyDictionary(network_string):
        return MockSkyDictionary(network_string)

class MockLikelihood:
    """Mock likelihood module."""
    
    marginalization = MockLikelihoodMarginalization()

class MockUtils:
    """Mock utils module."""
    
    @staticmethod
    def real_matmul(a, b):
        """Mock real matrix multiplication."""
        return np.real(np.matmul(a, b))

class MockCogwheel:
    """Main mock cogwheel module."""
    
    likelihood = MockLikelihood()
    utils = MockUtils()

# Create module-level attributes to mimic the real cogwheel structure
likelihood = MockLikelihood()
utils = MockUtils()