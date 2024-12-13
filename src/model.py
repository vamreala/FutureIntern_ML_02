import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib  # Use joblib for saving and loading models

# Load the data
train_data = pd.read_csv("train_data.csv")
test_data = pd.read_csv("test_data.csv")

# Separate features and labels
X_train = train_data.iloc[:, :-1]  # All columns except 'species'
y_train = train_data['species']

X_test = test_data.iloc[:, :-1]
y_test = test_data['species']

# Select a classification model
print("Training the model...")
# Uncomment the model you want to use
# model = LogisticRegression(max_iter=200)  # Logistic Regression
# model = DecisionTreeClassifier()  # Decision Tree
model = SVC()  # Support Vector Machine (SVM)

# Train the model
model.fit(X_train, y_train)
print("Model training complete.")

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

print("\nModel Performance:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, "iris_model.pkl")
print("\nModel saved as iris_model.pkl")