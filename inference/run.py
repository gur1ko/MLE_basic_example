"""
Script loads the latest trained PyTorch model, data for inference, and predicts results.
"""

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

# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))


CONF_FILE = os.path.join(ROOT_DIR, "../settings.json")

from utils import get_project_dir, configure_logging

# Loads configuration settings from JSON
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Defines paths
DATA_DIR = get_project_dir(conf['general']['data_dir'])
MODEL_DIR = get_project_dir(conf['general']['models_dir'])
RESULTS_DIR = get_project_dir(conf['general']['results_dir'])

# Initializes parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--infer_file", 
                    help="Specify inference data file", 
                    default=conf['inference']['inp_table_name'])
parser.add_argument("--out_path", 
                    help="Specify the path to the output table")


class SimpleNN(nn.Module):
    """Defines the same neural network architecture as in train.py"""
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
    """Gets the path of the latest saved PyTorch model"""
    latest = None
    for (dirpath, dirnames, filenames) in os.walk(MODEL_DIR):
        for filename in filenames:
            if filename.endswith(".pth") and (not latest or datetime.strptime(latest, conf['general']['datetime_format']) < 
                                              datetime.strptime(filename.split('.')[0], conf['general']['datetime_format'])):
                latest = filename
    return os.path.join(MODEL_DIR, latest)


def load_model(path: str, input_size: int, hidden_size: int, num_classes: int) -> nn.Module:
    """Loads and returns the PyTorch model"""
    model = SimpleNN(input_size, hidden_size, num_classes)
    model.load_state_dict(torch.load(path))
    model.eval()
    logging.info(f'Model loaded from {path}')
    return model


def preprocess_data(path: str, feature_columns: list) -> torch.Tensor:
    """Loads and preprocesses the inference data"""
    try:
        df = pd.read_csv(path)
        X = df[feature_columns].values

        # Scaling features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Convert to tensor
        return torch.tensor(X, dtype=torch.float32), df
    except Exception as e:
        logging.error(f"An error occurred while loading or preprocessing data: {e}")
        sys.exit(1)


def predict_results(model: nn.Module, data: torch.Tensor) -> list:
    """Predicts the results using the PyTorch model"""
    with torch.no_grad():
        outputs = model(data)
        predictions = torch.argmax(outputs, dim=1).tolist()
    return predictions


def store_results(df: pd.DataFrame, predictions: list, path: str = None) -> None:
    """Stores the prediction results in the results directory"""
    df["predictions"] = predictions
    if not path:
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
        path = os.path.join(RESULTS_DIR, datetime.now().strftime(conf['general']['datetime_format']) + '.csv')
    df.to_csv(path, index=False)
    logging.info(f'Results saved to {path}')


def main():
    """Main function"""
    configure_logging()
    args = parser.parse_args()

    # Model and data configurations
    feature_columns = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
    input_size = len(feature_columns)
    hidden_size = 16
    num_classes = 3

    # Load the latest model
    model_path = get_latest_model_path()
    model = load_model(model_path, input_size, hidden_size, num_classes)

    # Preprocess the data
    infer_file = args.infer_file
    data, df = preprocess_data(os.path.join(DATA_DIR, infer_file), feature_columns)

    # Make predictions
    predictions = predict_results(model, data)

    # Save results
    store_results(df, predictions, args.out_path)

    logging.info("Inference completed successfully.")


if __name__ == "__main__":
    main()
