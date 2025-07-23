
import numpy as np
from adapters.scratch.bagging_adf import BaggingADF

# --- Simulated Training Phase ---
model = BaggingADF(ctx_dim=2, act_dim=2, num_models=5, learning_rate=0.1)

interactions = [
    (np.array([0.1, 0.2]), np.array([1.0, 0.5]), 1.0, 0.5),
    (np.array([0.3, 0.1]), np.array([0.2, 0.8]), 0.0, 0.6),
    (np.array([0.2, 0.3]), np.array([0.9, 0.7]), 0.8, 0.4)
]

for ctx, act, r, p in interactions:
    _, done = model.update(ctx, act, r, p)

model.save(name="bagging_adf")

# --- Prediction Before Loading ---
new_model = BaggingADF(ctx_dim=2, act_dim=2)
test_context = np.array([0.2, 0.1])
test_actions = [np.array([1.0, 0.5]), np.array([0.2, 0.8]), np.array([0.9, 0.7])]

print("Predictions BEFORE loading model:")
print(new_model.predict(test_context, test_actions))

# --- Prediction After Loading ---
new_model.load(name="bagging_adf")
print("Predictions AFTER loading model:")
print(new_model.predict(test_context, test_actions))

print("Choice AFTER loading model:")
print(new_model.choose(test_context, test_actions))
