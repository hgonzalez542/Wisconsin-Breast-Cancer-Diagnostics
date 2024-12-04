import pandas as pd

# Define column names based on dataset description
columns = [
    "ID", "Diagnosis",
    "Mean Radius", "Mean Texture", "Mean Perimeter", "Mean Area", "Mean Smoothness",
    "Mean Compactness", "Mean Concavity", "Mean Concave Points", "Mean Symmetry", "Mean Fractal Dimension",
    "Radius SE", "Texture SE", "Perimeter SE", "Area SE", "Smoothness SE",
    "Compactness SE", "Concavity SE", "Concave Points SE", "Symmetry SE", "Fractal Dimension SE",
    "Worst Radius", "Worst Texture", "Worst Perimeter", "Worst Area", "Worst Smoothness",
    "Worst Compactness", "Worst Concavity", "Worst Concave Points", "Worst Symmetry", "Worst Fractal Dimension"
]

# Load dataset
data = pd.read_csv("wdbc.data", header=None, names=columns)

# Clean the dataset
data.drop(columns=["ID"], inplace=True)  # Remove ID column
data["Diagnosis_Numeric"] = data["Diagnosis"].map({"M": 1, "B": 0})  # Encode target as numeric

# Display first few rows of cleaned data
print(data.head())