

```markdown
# Iris Classification Project

This project demonstrates a structured approach to building and deploying a machine learning pipeline for classifying Iris flowers using PyTorch. It includes scripts for data generation, model training, and batch inference.

## Prerequisites

1. **Install Python (3.10 or higher)**:
   Download and install Python from [python.org](https://www.python.org/).

2. **Install Required Libraries**:
   Use the `requirements.txt` file to install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install MLFlow**:
   Install MLFlow for experiment tracking:
   ```bash
   pip install mlflow
   ```

4. **Install Docker (Optional)**:
   Docker is required to run training and inference in containers. Download Docker Desktop from [Docker’s official site](https://www.docker.com/products/docker-desktop).

## Project Structure

```
ML_example
├── data                      # Data files for training and inference
│   ├── iris_inference.csv
│   ├── iris_train.csv
├── data_process              # Scripts for data generation
│   ├── data_generation.py
├── inference                 # Inference scripts and Dockerfile
│   ├── Dockerfile
│   ├── run.py
├── models                    # Trained models
│   ├── simple_nn.pth
├── results                   # Inference results
│   ├── inference_results.csv
├── training                  # Training scripts and Dockerfile
│   ├── Dockerfile
│   ├── train.py
├── tests                     # Unit tests for project components
│   ├── test_data_generation.py
│   ├── test_train.py
│   ├── test_inference.py
├── settings.json             # Configuration settings
├── requirements.txt          # Dependency file
├── .gitignore                # Git ignore file
└── README.md                 # Project documentation
```

## How to Run

### 1. Generate Data
Run the data generation script to create training and inference datasets:
```bash
python data_process/data_generation.py
```

### 2. Train the Model
Train a PyTorch model and log metrics to MLFlow:
```bash
python training/train.py
```

### 3. Run Inference
Perform batch inference using the trained model:
```bash
python inference/run.py
```

### 4. View MLFlow UI
Start the MLFlow tracking server:
```bash
mlflow ui
```
Access it at `http://localhost:5000`.

### 5. Run Tests
Validate functionality with unit tests for:
- Data generation (`test_data_generation.py`)
- Model training (`test_train.py`)
- Batch inference (`test_inference.py`)

Run all tests:
```bash
pytest tests/

## Docker Integration (Optional)

**Note:** Docker integration has not been implemented or tested for this project due to installation issues. However, the following steps outline how Docker support could be added for training and inference:

#### Training
1. Create a `Dockerfile` in the `training` directory with the following content:
   ```dockerfile
   # Use a base image with Python
   FROM python:3.10-slim

   # Set the working directory
   WORKDIR /app

   # Copy files into the container
   COPY training/train.py .
   COPY requirements.txt .
   COPY settings.json .
   COPY models/ ./models
   COPY data/ ./data

   # Install dependencies
   RUN pip install --no-cache-dir -r requirements.txt

   # Command to run the training script
   CMD ["python", "train.py"]
   ```

2. Build the Docker image:
   ```bash
   docker build -f ./training/Dockerfile -t training_image .
   ```

3. Run the training container:
   ```bash
   docker run training_image
   ```

#### Inference
1. Create a `Dockerfile` in the `inference` directory with the following content:
   ```dockerfile
   # Use a base image with Python
   FROM python:3.10-slim

   # Set the working directory
   WORKDIR /app

   # Copy files into the container
   COPY inference/run.py .
   COPY requirements.txt .
   COPY settings.json .
   COPY models/ ./models
   COPY data/ ./data

   # Install dependencies
   RUN pip install --no-cache-dir -r requirements.txt

   # Command to run the inference script
   CMD ["python", "run.py"]
   ```

2. Build the Docker image:
   ```bash
   docker build -f ./inference/Dockerfile -t inference_image .
   ```

3. Run the inference container:
   ```bash
   docker run inference_image
   ```

**Future Work:** Docker can be revisited and implemented once installation issues are resolved.


## Insights
1. **Dataset**:
   - The Iris dataset was split into training (80%) and inference (20%) sets.

2. **Model**:
   - A simple neural network with one hidden layer was trained using PyTorch.
   - Training logs and model artifacts were saved to MLFlow.

3. **Performance**:
   - Loss values were logged during training for each epoch.

## Conclusion
This project demonstrates a modular approach to building ML pipelines. The integration of MLFlow and Docker ensures reproducibility, and the unit tests validate key components of the workflow.

### Potential Improvements:
- Extend the model to support hyperparameter tuning using MLFlow.
- Deploy the inference container to a cloud platform for scalability.

