import pandas as pd
import logging
import os
import json
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from utils import singleton, get_project_dir, configure_logging

# Configure logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONF_FILE = os.path.join(ROOT_DIR, "../settings.json")

with open(CONF_FILE, "r") as file:
    conf = json.load(file)

DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, '../data'))
TRAIN_PATH = os.path.join(DATA_DIR, conf["train"]["table_name"])
INFERENCE_PATH = os.path.join(DATA_DIR, conf["inference"]["inp_table_name"])

# Ensure the data directory exists
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    logger.info(f"Created directory: {DATA_DIR}")


@singleton
class IrisDataGenerator:
    def __init__(self):
        self.df = None

    def create(self, train_path: str, inference_path: str, test_size: float = 0.2, random_state: int = 42):
        """
        Generates training and inference datasets from the Iris dataset.
        """
        logger.info("Loading Iris dataset...")
        iris = load_iris(as_frame=True)
        df = iris.frame

        logger.info("Splitting dataset into training and inference sets...")
        train_df, inference_df = train_test_split(
            df, test_size=test_size, random_state=random_state
        )

        logger.info("Saving datasets...")
        self.save(train_df, train_path)
        self.save(inference_df, inference_path)

    def save(self, df: pd.DataFrame, out_path: str):
        """
        Saves a DataFrame to the specified path, ensuring the directory exists.
        """
        directory = os.path.dirname(out_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")
        
        logger.info(f"Saving data to {out_path}...")
        df.to_csv(out_path, index=False)


if __name__ == "__main__":
    configure_logging()
    logger.info("Starting data generation script...")
    generator = IrisDataGenerator()
    generator.create(train_path=TRAIN_PATH, inference_path=INFERENCE_PATH)
    logger.info("Data generation script completed successfully.")
