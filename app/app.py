import pickle
import numpy as np
from src.genai_advisor import generate_health_advice

# Load model and scaler
with open("models/diabetes_model.pkl", "rb") as f:
    model, scaler = pickle.load(f)

sample_input = np.array([[120, 70, 30.5, 25, 150, 0, 1, 33]])
sample_input = scaler.transform(sample_input)

prediction = model.predict(sample_input)[0]
advice = generate_health_advice(prediction)

print("Prediction:", "Diabetic" if prediction == 1 else "Non-Diabetic")
print("\nAI Health Advice:\n", advice)
