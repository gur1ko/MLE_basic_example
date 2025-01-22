import argparse
import json
import logging
import os
import sys
from datetime import datetime
import torch
from torch import nn
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import get_project_dir, configure_logging

# Load configuration settings
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONF_FILE = os.path.join(ROOT_DIR, "../settings.json")

with open(CONF_FILE, "r") as file:
    conf = json.load(file)

DATA_DIR = get_project_dir(conf["general"]["data_dir"])
MODEL_DIR = get_project_dir(conf["general"]["models_dir"])
RESULTS_DIR = get_project_dir(conf["general"]["results_dir"])

# Argument parser for command-line inputs
parser = argparse.ArgumentParser()
parser.add_argument("--infer_file", default=conf["inference"]["inp_table_name"], help="Inference data file")
parser.add_argument("--out_path", help="Output file path for results")


class SimpleNN(nn.Module):
    """Defines the neural network architecture."""
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_latest_model_path() -> str:
    """Finds the path to the most recently saved model."""
    latest = None
    for dirpath, _, filenames in os.walk(MODEL_DIR):
        for filename in filenames:
            if filename.endswith(".pth") and (
                not latest or datetime.strptime(latest, conf["general"]["datetime_format"]) <
                datetime.strptime(filename.split('.')[0], conf["general"]["datetime_format"])
            ):
                latest = filename
    return os.path.join(MODEL_DIR, latest)


def load_model(path: str, input_size: int, hidden_size: int, num_classes: int) -> nn.Module:
    """Loads the PyTorch model from the given path."""
    model = SimpleNN(input_size, hidden_size, num_classes)
    model.load_state_dict(torch.load(path))
    model.eval()
    logging.info(f"Model loaded from {path}")
    return model


def preprocess_data(path: str, feature_columns: list) -> tuple:
    """Prepares the inference dataset."""
    try:
        df = pd.read_csv(path)
        X = df[feature_columns].values

        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Convert to tensor
        return torch.tensor(X, dtype=torch.float32), df
    except Exception as e:
        logging.error(f"Error during data preprocessing: {e}")
        sys.exit(1)


def predict_results(model: nn.Module, data: torch.Tensor) -> list:
    """Generates predictions using the trained model."""
    with torch.no_grad():
        outputs = model(data)
        return torch.argmax(outputs, dim=1).tolist()


def store_results(df: pd.DataFrame, predictions: list, path: str = None) -> None:
    """Saves the predictions to a results file."""
    df["predictions"] = predictions
    if not path:
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
        path = os.path.join(RESULTS_DIR, datetime.now().strftime(conf["general"]["datetime_format"]) + ".csv")
    df.to_csv(path, index=False)
    logging.info(f"Results saved to {path}")


def run_inference():
    """Executes the inference process."""
    configure_logging()

    feature_columns = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
    input_size = len(feature_columns)
    hidden_size = 16
    num_classes = 3

    # Load model
    model_path = get_latest_model_path()
    model = load_model(model_path, input_size, hidden_size, num_classes)

    # Preprocess data
    infer_file = os.path.join(DATA_DIR, conf["inference"]["inp_table_name"])
    data, df = preprocess_data(infer_file, feature_columns)

    # Predict and save results
    predictions = predict_results(model, data)
    store_results(df, predictions)

    logging.info("Inference completed successfully.")


if __name__ == "__main__":
    args = parser.parse_args()
    run_inference()
