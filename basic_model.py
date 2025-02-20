# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Load the Breast Cancer dataset
cancer = datasets.load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = pd.Series(cancer.target)

# Display basic information about the dataset
print("Dataset Information:")
print(X.info())
print("\nClass Labels:", cancer.target_names)
print("\nFirst 5 rows of the dataset:")
print(X.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=cancer.target_names)
conf_matrix = confusion_matrix(y_test, y_pred)

# Output accuracy and classification report
print("\nModel Accuracy:", accuracy)
print("\nClassification Report:\n", report)

# Visualize the Decision Tree
plt.figure(figsize=(20,10))
plot_tree(model, feature_names=cancer.feature_names, class_names=cancer.target_names, filled=True)
plt.title("Decision Tree Visualization - Breast Cancer Dataset")
plt.show()

# Confusion Matrix Visualization
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=cancer.target_names, yticklabels=cancer.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
