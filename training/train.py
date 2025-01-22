"""
This script prepares the data, trains a simple PyTorch model, and saves it.
"""

import os
import sys
import pickle
import json
import logging
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Load configuration
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONF_FILE = os.path.join(ROOT_DIR, "../settings.json")

with open(CONF_FILE, "r") as file:
    conf = json.load(file)

DATA_DIR = os.path.join(ROOT_DIR, "../data")
MODEL_DIR = os.path.join(ROOT_DIR, "../models")
TRAIN_PATH = os.path.join(DATA_DIR, conf["train"]["table_name"])

# Logging setup
logging.basicConfig(level=logging.INFO)

# Define the model
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Main training function
def train_model():
    logging.info("Loading data...")
    df = pd.read_csv(TRAIN_PATH)

    # Preprocessing
    logging.info("Preprocessing data...")
    feature_columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    label_column = "target"


    # Encode labels
    le = LabelEncoder()
    df[label_column] = le.fit_transform(df[label_column])

    # Split data
    X = df[feature_columns].values
    y = df[label_column].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=conf["train"]["test_size"], random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to tensors
    train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_data = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=32, shuffle=False)

    # Initialize the model
    input_size = len(feature_columns)
    hidden_size = 16
    num_classes = len(df[label_column].unique())
    model = SimpleNN(input_size, hidden_size, num_classes)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    logging.info("Training the model...")
    for epoch in range(10):  # Train for 10 epochs
        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        logging.info(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    # Save the model
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    model_path = os.path.join(MODEL_DIR, "simple_nn.pth")
    torch.save(model.state_dict(), model_path)
    logging.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    train_model()
