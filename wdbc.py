import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load and clean the dataset
columns = ["Diagnosis", "Radius Mean", "Perimeter Mean", "Area Mean"]
data = pd.read_csv("data.csv", skiprows=1, header=None, names=columns)

# Convert Diagnosis to a categorical type for better visualization
data["Diagnosis"] = data["Diagnosis"].map({"M": "Malignant", "B": "Benign"})

# Verify dataset
print("Dataset preview:")
print(data.head())

# Function to create scatter plots for better visualization
def create_scatterplot(x_feature, y_feature):
    plt.figure(figsize=(8, 6))
    
    # Create a scatter plot with Malignant (red) and Benign (blue) coloring
    sns.scatterplot(x=x_feature, y=y_feature, hue="Diagnosis", data=data, palette={"Malignant": "red", "Benign": "blue"}, alpha=0.8)

    # Adding title and labels
    plt.title(f"{y_feature} vs {x_feature}")
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)

    # Adding custom legend
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, labels, title="Diagnosis", loc="upper left")

    # Display the plot
    plt.tight_layout()
    plt.show()

# Scatter plots for feature relationships with colored dots
create_scatterplot("Radius Mean", "Perimeter Mean")
create_scatterplot("Radius Mean", "Area Mean")
create_scatterplot("Perimeter Mean", "Area Mean")
