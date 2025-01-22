import os
import sys

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.train import train_model

def test_training():
    """
    Tests the training process by verifying:
    1. The model file is created.
    """
    train_model()
    model_path = "../models/simple_nn.pth"
    assert os.path.exists(model_path), f"Model file not found: {model_path}"
    print("Test passed: Model file exists!")


if __name__ == "__main__":
    test_training()
