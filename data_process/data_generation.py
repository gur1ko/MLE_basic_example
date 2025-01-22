
# Importing required libraries
import numpy as np
import pandas as pd
import logging
import os
import sys
import json
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Define directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))
from utils import singleton, get_project_dir, configure_logging

DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, '../data'))
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Change to CONF_FILE = "settings.json" if you have problems with env variables
CONF_FILE = os.path.join(ROOT_DIR, "../settings.json")

# Load configuration settings from JSON
logger.info("Loading configuration settings from JSON...")
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Define paths
logger.info("Defining paths...")
DATA_DIR = get_project_dir(conf['general']['data_dir'])
TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['table_name'])
INFERENCE_PATH = os.path.join(DATA_DIR, conf['inference']['inp_table_name'])

# Singleton class for generating Iris dataset
@singleton
class IrisDataGenerator():
    def __init__(self):
        self.df = None

    # Method to load and process the Iris dataset
    def create(self, train_path: str, inference_path: str, test_size: float = 0.2, random_state: int = 42):
        logger.info("Loading Iris dataset...")
        iris = load_iris(as_frame=True)
        df = iris.frame

        logger.info("Splitting dataset into training and inference sets...")
        train_df, inference_df = train_test_split(df, test_size=test_size, random_state=random_state)

        logger.info("Saving datasets...")
        self.save(train_df, train_path)
        self.save(inference_df, inference_path)

    # Method to save data
    def save(self, df: pd.DataFrame, out_path: str):
        logger.info(f"Saving data to {out_path}...")
        df.to_csv(out_path, index=False)

# Main execution
if __name__ == "__main__":
    configure_logging()
    logger.info("Starting script...")
    gen = IrisDataGenerator()
    gen.create(train_path=TRAIN_PATH, inference_path=INFERENCE_PATH)
    logger.info("Script completed successfully.")
