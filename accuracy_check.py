import pandas as pd

# Load the results
results_path = "c:/Users/gurami.kiknadze/Desktop/EPAM HOMEWORK LAST1/ML_example/results/22.01.2025_22.09.csv"
df = pd.read_csv(results_path)

# Calculate accuracy
correct_predictions = (df['target'] == df['predictions']).sum()
total_predictions = len(df)
accuracy = correct_predictions / total_predictions

print(f"Model Accuracy: {accuracy:.2%}")
