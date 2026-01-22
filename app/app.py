import pickle
import numpy as np
from src.genai_advisor import generate_health_advice

# Load trained ML model
with open("models/diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

# Sample input data
sample_input = np.array([[120, 70, 30.5, 25, 150, 0, 1, 33]])

# Prediction
prediction = model.predict(sample_input)[0]

# Generate AI advice
advice = generate_health_advice(prediction)

print("Prediction:", "Diabetic" if prediction == 1 else "Non-Diabetic")
print("\nAI Health Advice:\n", advice)
