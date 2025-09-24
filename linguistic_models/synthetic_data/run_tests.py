#!/usr/bin/env python
"""
Convenience script to run tests for the synthetic data generation module.
"""

import subprocess
import sys
import os


def run_tests():
    """Run the test suite."""
    print("Running synthetic data generation tests...")
    
    # Change to the correct directory
    os.chdir(os.path.dirname(__file__))
    
    # Run pytest
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        "tests/unit/", 
        "-v",
        "--tb=short"
    ])
    
    if result.returncode == 0:
        print("All tests passed!")
    else:
        print("Some tests failed.")
        sys.exit(1)


def run_integration_tests():
    """Run only integration tests."""
    print("Running integration tests...")
    
    os.chdir(os.path.dirname(__file__))
    
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        "tests/unit/", 
        "-v",
        "-m", "integration"
    ])
    
    if result.returncode == 0:
        print("All integration tests passed!")
    else:
        print("Some integration tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "integration":
        run_integration_tests()
    else:
        run_tests()