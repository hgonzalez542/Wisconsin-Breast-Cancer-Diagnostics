import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Define column names based on dataset description
columns = [
    "Diagnosis", 
    "Mean Radius", "Mean Perimeter", "Mean Area", 
    "Radius SE", "Perimeter SE", "Area SE", 
    "Worst Radius", "Worst Perimeter", "Worst Area"
]

# Load dataset
data = pd.read_csv("data.csv", header=None, names=columns)

# Check for missing values
print(f"Missing values in dataset:\n{data.isnull().sum()}")

# Remove non-numeric characters (if any) in numeric columns
def clean_column(col):
    # Remove any non-numeric characters (e.g., 'radius_mean17.99' -> '17.99')
    return col.replace(r'[^\d.]', '', regex=True)

# Clean columns that should be numeric
data["Mean Radius"] = data["Mean Radius"].apply(clean_column)
data["Mean Perimeter"] = data["Mean Perimeter"].apply(clean_column)
data["Mean Area"] = data["Mean Area"].apply(clean_column)
data["Worst Radius"] = data["Worst Radius"].apply(clean_column)
data["Worst Perimeter"] = data["Worst Perimeter"].apply(clean_column)
data["Worst Area"] = data["Worst Area"].apply(clean_column)

# Convert the columns to numeric
data["Mean Radius"] = pd.to_numeric(data["Mean Radius"], errors="coerce")
data["Mean Perimeter"] = pd.to_numeric(data["Mean Perimeter"], errors="coerce")
data["Mean Area"] = pd.to_numeric(data["Mean Area"], errors="coerce")
data["Worst Radius"] = pd.to_numeric(data["Worst Radius"], errors="coerce")
data["Worst Perimeter"] = pd.to_numeric(data["Worst Perimeter"], errors="coerce")
data["Worst Area"] = pd.to_numeric(data["Worst Area"], errors="coerce")

# After cleaning, fill missing values with the mean or median
data.fillna(data.mean(), inplace=True)  # You can replace 'data.mean()' with 'data.median()' if preferred

# Map Diagnosis to numeric (1 for Malignant, 0 for Benign)
data["Diagnosis_Numeric"] = data["Diagnosis"].map({"M": 1, "B": 0})

# Display the first few rows of the dataset to ensure it's correct
print(data.head())

# Set global style and context for cleaner plots
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.3)

# Scatter Plot: Relationship Between Mean Perimeter and Mean Area
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=data, 
    x="Mean Perimeter", 
    y="Mean Area", 
    hue="Diagnosis",  # This will color the dots based on diagnosis (Malignant vs Benign)
    palette="coolwarm",  # coolwarm gives distinct colors for categories
    alpha=0.7,  # Slight transparency for better visibility when points overlap
    s=100  # Size of the points in the scatter plot
)
plt.title("Relationship Between Mean Perimeter and Mean Area by Diagnosis", fontsize=16)
plt.xlabel("Mean Perimeter", fontsize=14)
plt.ylabel("Mean Area", fontsize=14)
plt.legend(title="Diagnosis", fontsize=12)
plt.tight_layout()
plt.show()  # Ensure only one show() call for scatter plot

# Correlation Heatmap for Selected Features
selected_features = [
    "Mean Radius", "Mean Perimeter", "Mean Area", 
    "Worst Radius", "Worst Perimeter", "Worst Area", 
    "Diagnosis_Numeric"
]

plt.figure(figsize=(10, 8))
sns.heatmap(
    data[selected_features].corr(), 
    annot=True, 
    cmap="coolwarm", 
    fmt=".2f", 
    linewidths=0.5
)
plt.title("Correlation Heatmap of Selected Features", fontsize=18)
plt.tight_layout()
plt.show()  # Ensure only one show() call for the heatmap

# Logistic Regression Model (Optional)
# Select features for model
X = data[["Worst Radius", "Worst Area", "Worst Perimeter"]]
y = data["Diagnosis_Numeric"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))
