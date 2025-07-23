
import numpy as np
from adapters.scratch.regcb_elim import RegCBElim

# --- Simulated Training Phase ---
model = RegCBElim(ctx_dim=2, act_dim=2, alpha=1.0)

interactions = [
    (np.array([0.1, 0.2]), np.array([1.0, 0.5]), 1.0, 0.5),
    (np.array([0.3, 0.1]), np.array([0.2, 0.8]), 0.0, 0.6),
    (np.array([0.2, 0.3]), np.array([0.9, 0.7]), 0.8, 0.4)
]

for ctx, act, r, p in interactions:
    _, done = model.update(ctx, act, r, p)

model.save(name="regcb_elim")

# --- Prediction Before Loading ---
new_model = RegCBElim(ctx_dim=2, act_dim=2, alpha=1.0)
test_context = np.array([0.2, 0.1])
test_actions = [np.array([1.0, 0.5]), np.array([0.2, 0.8]), np.array([0.9, 0.7])]

print("Surviving Actions BEFORE loading model:")
print(new_model.predict(test_context, test_actions))

# --- Prediction After Loading ---
new_model.load(name="regcb_elim")
print("Surviving Actions AFTER loading model:")
print(new_model.predict(test_context, test_actions))

print("Chosen Action AFTER loading model:")
print(new_model.choose(test_context, test_actions))
