
import numpy as np
from adapters.scratch.linucb_adapter import LinUCBModel

# --- Simulated Training Phase ---
model = LinUCBModel(ctx_dim=2, act_dim=2, alpha=0.5)

interactions = [
    (np.array([0.1, 0.2]), np.array([1.0, 0.5]), 1.0, 0.5),
    (np.array([0.3, 0.1]), np.array([0.2, 0.8]), 0.0, 0.6),
    (np.array([0.2, 0.3]), np.array([0.9, 0.7]), 0.8, 0.4)
]

for ctx, act, r, p in interactions:
    _, done = model.update(ctx, act, r)

model.save(name="lin_ucb")

# --- Prediction Before Loading ---
new_model = LinUCBModel(ctx_dim=2, act_dim=2, alpha=0.5)
test_context = np.array([0.2, 0.1])
test_actions = [np.array([1.0, 0.5]), np.array([0.2, 0.8]), np.array([0.9, 0.7])]

print("Choice BEFORE loading model:")
print(new_model.predict(test_context, test_actions, alpha=0.5))

# --- Prediction After Loading ---
new_model.load(name="lin_ucb")
print("Choice AFTER loading model:")
print(new_model.predict(test_context, test_actions, alpha=0.5))
