import numpy as np
from adapters.scratch.neural_squarecb import NeuralSquareCB

# --- Simulated Training Phase ---
model = NeuralSquareCB(ctx_dim=2, act_dim=2, hidden_dims=[64, 32], gamma=100.0)

interactions = [
    (np.array([0.1, 0.2]), np.array([1.0, 0.5]), 1.0, 0.5),
    (np.array([0.3, 0.1]), np.array([0.2, 0.8]), 0.0, 0.6),
    (np.array([0.2, 0.3]), np.array([0.9, 0.7]), 0.8, 0.4)
]

for ctx, act, r, p in interactions:
    _, done = model.update(ctx, act, r, p)

model.save(name="neural_squarecb")

# --- Prediction Before Loading ---
new_model = NeuralSquareCB(ctx_dim=2, act_dim=2, hidden_dims=[64, 32], gamma=100.0)
test_context = np.array([0.2, 0.1])
test_actions = [np.array([1.0, 0.5]), np.array([0.2, 0.8]), np.array([0.9, 0.7])]

print("Predictions BEFORE loading model:")
print(new_model.predict(test_context, test_actions))

# --- Prediction After Loading ---
new_model.load(name="neural_squarecb")
print("Predictions AFTER loading model:")
print(new_model.predict(test_context, test_actions))

print("Choice AFTER loading model:")
print(new_model.choose(test_context, test_actions))
