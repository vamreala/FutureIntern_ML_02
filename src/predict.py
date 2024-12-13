import joblib
import numpy as np

# Load the trained model
model = joblib.load("iris_model.pkl")

# Mapping of numeric predictions to species names
species_mapping = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}

# Example input: Sepal Length, Sepal Width, Petal Length, Petal Width
input_data = [[5.1, 3.5, 1.4, 0.2]]  # Adjust this as per your input

# Print the input to check it's correctly formatted
print("Input Data:", input_data)

# Make a prediction
prediction = model.predict(input_data)

# Print the raw prediction (numeric output)
print("Raw Prediction (numeric):", prediction)

# Convert numeric prediction to species name
predicted_species = species_mapping[prediction[0]]

# Output the species name
print(f"Prediction for the input {input_data}: {predicted_species}")