import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = load_iris()
data = pd.DataFrame(
    data=iris.data,
    columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
)
data['species'] = iris.target

# Display dataset preview
print("Dataset loaded successfully. Here's a preview:")
print(data.head())

# Split the data into training and testing sets
X = data.iloc[:, :-1]
y = data['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Combine features and target for train and test sets
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

# Save the datasets
train_data.to_csv("train_data.csv", index=False)
test_data.to_csv("test_data.csv", index=False)

print("\nData split into training and testing sets and saved as CSV files.")
print(f"Training data shape: {train_data.shape}")
print(f"Testing data shape: {test_data.shape}")
