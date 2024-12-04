import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
data = pd.read_csv("data.csv", header=None, names=columns)

# Clean the dataset
data.drop(columns=["ID"], inplace=True)  # Remove ID column
data["Diagnosis_Numeric"] = data["Diagnosis"].map({"M": 1, "B": 0})  # Encode target as numeric

# Display first few rows of cleaned data
print(data.head())

# Visualize the distribution of selected features by diagnosis
features = ["Mean Radius", "Mean Perimeter", "Mean Area", "Worst Radius", "Worst Perimeter", "Worst Area"]
for feature in features:
    plt.figure(figsize=(8, 6))
    sns.histplot(data, x=feature, hue="Diagnosis", kde=True, palette="Set2", bins=30)
    plt.title(f"{feature} Distribution by Diagnosis")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.show()

# Correlation heatmap for selected features and target variable
selected_features = features + ["Diagnosis_Numeric"]
plt.figure(figsize=(8, 6))
sns.heatmap(data[selected_features].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
