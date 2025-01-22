import os
import sys
import pandas as pd

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_process.data_generation import IrisDataGenerator

def test_data_generation():
    generator = IrisDataGenerator()
    generator.create(
        train_path="../data/iris_train.csv",
        inference_path="../data/iris_inference.csv"
    )
    assert os.path.exists("../data/iris_train.csv")
    assert os.path.exists("../data/iris_inference.csv")

    train_data = pd.read_csv("../data/iris_train.csv")
    inference_data = pd.read_csv("../data/iris_inference.csv")
    assert not train_data.empty
    assert not inference_data.empty

    print("Test passed: Data generation works as expected!")

# Run the test when the script is executed directly
if __name__ == "__main__":
    test_data_generation()
