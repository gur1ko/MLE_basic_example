import os
import sys

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference.run import run_inference

def test_inference():
    """
    Tests the inference process by verifying:
    1. The inference results file is created.
    """
    run_inference()
    results_path = "../results/inference_results.csv"
    assert os.path.exists(results_path), f"Results file not found: {results_path}"
    print("Test passed: Inference results file exists!")


if __name__ == "__main__":
    test_inference()
