from tensorflow.keras.models import load_model
import numpy as np

model = load_model("models/asl_model.h5")
print("Model Summary:")
model.summary()
print("\nInput Shape:", model.input_shape)
print("Output Shape:", model.output_shape)

# Try a test prediction
test_input = np.zeros((1, 63), dtype=np.float32)
print("\nTest prediction with shape (1, 63):")
try:
    pred = model.predict(test_input, verbose=0)
    print("SUCCESS! Prediction shape:", pred.shape)
except Exception as e:
    print("FAILED:", e)
    
# If that fails, try with sequence dimension
test_input_seq = np.zeros((1, 1, 63), dtype=np.float32)
print("\nTest prediction with shape (1, 1, 63):")
try:
    pred = model.predict(test_input_seq, verbose=0)
    print("SUCCESS! Prediction shape:", pred.shape)
except Exception as e:
    print("FAILED:", e)