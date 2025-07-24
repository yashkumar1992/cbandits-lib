#!/usr/bin/env python3
# scripts/train_tf_agents_cb.py

import numpy as np
import sys
import os

# Add parent directory to path
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adapters.tensorflow.tf_agents_cb import TFAgentsCB


print("🚀 Starting TF-Agents Contextual Bandit Training Demo")
print("=" * 60)

# --- Test LinUCB Agent ---
print("\n🔧 Testing LinUCB Agent")
print("-" * 30)

model_linucb = TFAgentsCB(ctx_dim=2, act_dim=2, num_actions=3, agent_type='neural_lin_ucb', alpha=1.0)

# Same interactions as lin_reg_iwr
interactions = [
    (np.array([0.1, 0.2]), np.array([1.0, 0.5]), 1.0, 0.5),
    (np.array([0.3, 0.1]), np.array([0.2, 0.8]), 0.0, 0.6),
    (np.array([0.2, 0.3]), np.array([0.9, 0.7]), 0.8, 0.4)
]

print("🎯 Training LinUCB model...")
for i, (ctx, act, r, p) in enumerate(interactions):
    result, done = model_linucb.update(ctx, act, r, p)
    print(f"  Step {i+1}: {result['status']}")

print("💾 Saving LinUCB model...")
save_path = model_linucb.save(name="tf_agents_linucb")
print(f"✅ LinUCB model saved")

# --- Test Neural Epsilon-Greedy Agent ---
print("\n🔧 Testing Neural Epsilon-Greedy Agent")
print("-" * 40)

model_neural = TFAgentsCB(
    ctx_dim=2, act_dim=2, num_actions=3, 
    agent_type='neural_epsilon_greedy', 
    learning_rate=0.01
)

print("🎯 Training Neural model...")
for i, (ctx, act, r, p) in enumerate(interactions):
    result, done = model_neural.update(ctx, act, r, p)
    print(f"  Step {i+1}: {result['status']}")

print("💾 Saving Neural model...")
save_path = model_neural.save(name="tf_agents_neural")
print(f"✅ Neural model saved")

# --- Test Predictions Before Loading ---
print("\n🔮 Testing Predictions BEFORE Loading")
print("-" * 40)

new_model = TFAgentsCB(ctx_dim=2, act_dim=2, num_actions=3, agent_type='lin_ucb')
test_context = np.array([0.2, 0.1])
test_actions = [np.array([1.0, 0.5]), np.array([0.2, 0.8]), np.array([0.9, 0.7])]

print("LinUCB predictions BEFORE loading:")
predictions_before = new_model.predict(test_context, test_actions)
print(f"  Predictions: {predictions_before}")
choice_before = new_model.choose(test_context, test_actions)
print(f"  Choice: {choice_before}")

# --- Test Predictions After Loading ---
print("\n🔄 Testing Predictions AFTER Loading")
print("-" * 40)

print("Loading LinUCB model...")
new_model.load(name="tf_agents_linucb")

print("LinUCB predictions AFTER loading:")
predictions_after = new_model.predict(test_context, test_actions)
print(f"  Predictions: {predictions_after}")
choice_after = new_model.choose(test_context, test_actions)
print(f"  Choice: {choice_after}")

# --- Test Neural Model Loading ---
print("\n🧠 Testing Neural Model Loading")
print("-" * 35)

neural_model = TFAgentsCB(ctx_dim=2, act_dim=2, num_actions=3, agent_type='neural_epsilon_greedy')
neural_model.load(name="tf_agents_neural")

print("Neural model predictions AFTER loading:")
neural_predictions = neural_model.predict(test_context, test_actions)
print(f"  Predictions: {neural_predictions}")
neural_choice = neural_model.choose(test_context, test_actions)
print(f"  Choice: {neural_choice}")

# --- Compare All Models ---
print("\n📊 Model Comparison")
print("-" * 25)
print(f"Test Context: {test_context}")
print(f"Test Actions: {[f'Action {i}: {action}' for i, action in enumerate(test_actions)]}")
print()
print("Results:")
print(f"  LinUCB Choice:     {choice_after}")
print(f"  Neural Choice:     {neural_choice}")
print(f"  LinUCB Predictions: {[f'{p:.3f}' for p in predictions_after]}")
print(f"  Neural Predictions: {[f'{p:.3f}' for p in neural_predictions]}")

# --- Extended Training Example ---
print("\n🔄 Extended Training Example")
print("-" * 35)

# Create more training data
extended_interactions = []
np.random.seed(42)

for _ in range(10):
    ctx = np.random.uniform(0, 1, 2)
    act = np.random.uniform(0, 1, 2)
    # Simulate reward based on context-action interaction
    reward = np.dot(ctx, act) + 0.1 * np.random.randn()
    prob = 1.0 / 3  # Uniform random policy
    extended_interactions.append((ctx, act, reward, prob))

print("Training LinUCB on extended dataset...")
extended_model = TFAgentsCB(ctx_dim=2, act_dim=2, num_actions=3, agent_type='lin_ucb', alpha=0.5)

for i, (ctx, act, r, p) in enumerate(extended_interactions):
    result, done = extended_model.update(ctx, act, r, p)
    if i % 3 == 0:
        print(f"  Processed {i+1}/{len(extended_interactions)} interactions")

print("Testing extended model predictions...")
extended_predictions = extended_model.predict(test_context, test_actions)
extended_choice = extended_model.choose(test_context, test_actions)

print(f"Extended model predictions: {[f'{p:.3f}' for p in extended_predictions]}")
print(f"Extended model choice: {extended_choice}")

# --- Performance Tips ---
print("\n💡 TF-Agents Tips & Best Practices")
print("-" * 40)
print("1. LinUCB is better for linear reward relationships")
print("2. Neural agents handle non-linear patterns but need more data")
print("3. Alpha parameter controls exploration in LinUCB (higher = more exploration)")
print("4. Neural networks need sufficient training data to converge")
print("5. TF-Agents handles arm features automatically in contextual bandits")

print("\n🎉 Demo completed successfully!")
