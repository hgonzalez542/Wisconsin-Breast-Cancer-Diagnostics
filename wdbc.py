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
    "Worst Radius", "Worst Perimeter", "Worst Area",
    
]

# Load dataset
data = pd.read_csv("data.csv", header=None, names=columns)



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

# Select features and target variable
X = data[["Worst Radius", "Worst Area", "Worst Texture"]]
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
