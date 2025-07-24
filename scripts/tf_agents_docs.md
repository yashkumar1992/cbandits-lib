# TF-Agents Contextual Bandit Integration

## Overview

TF-Agents is Google's reinforcement learning library built on TensorFlow. Our integration provides access to production-ready contextual bandit algorithms with arm features support.

## Supported Algorithms

### 1. Linear Upper Confidence Bound (LinUCB)
- **Use Case**: Linear reward relationships, fast convergence
- **Theory**: Uses ridge regression with confidence bounds for exploration
- **Math**: `reward = θᵀx + confidence_bonus` where confidence scales with uncertainty

### 2. Neural Epsilon-Greedy
- **Use Case**: Non-linear reward patterns, complex feature interactions  
- **Theory**: Neural network for reward prediction + ε-greedy exploration
- **Math**: `P(action) = ε/K + (1-ε)·δ(action = argmax_a Q(x,a))`

## Key Features

### ✅ Arm Features Support
- **Global Context**: User/environment features (e.g., demographics, time)
- **Per-Arm Context**: Action-specific features (e.g., content attributes)
- **Automatic Handling**: TF-Agents manages feature combination internally

### ✅ Production Ready
- **Distributed Training**: Built on TensorFlow's distributed computing
- **Model Serving**: Compatible with TensorFlow Serving
- **Scalability**: Handles large feature spaces and action sets

### ✅ Advanced Exploration
- **LinUCB**: Principled uncertainty-based exploration
- **Neural**: Configurable exploration strategies (ε-greedy, UCB, Thompson Sampling)

## Mathematical Foundation

### LinUCB Algorithm
```
For each action a:
1. Compute features: x_a = [global_context, arm_context_a]
2. Predict reward: μ_a = θᵀx_a  
3. Compute confidence: σ_a = √(x_a ᵀ A⁻¹ x_a)
4. Upper bound: UCB_a = μ_a + α·σ_a

Choose: a* = argmax_a UCB_a
Update: A += x_a x_a ᵀ, b += r·x_a, θ = A⁻¹b
```

### Neural Epsilon-Greedy
```
Network: f_θ(global_context, arm_features) → reward_prediction
Policy: π(a|x) = ε/|A| + (1-ε)·δ(a = argmax_a f_θ(x,a))
Update: θ ← θ - η·∇_θ (f_θ(x,a) - r)²
```

## Industry Best Practices

### Feature Engineering
1. **Normalization**: Scale features to [0,1] or standardize
2. **Categorical Encoding**: Use embeddings for high-cardinality categories
3. **Interaction Terms**: Let neural networks learn interactions automatically

### Hyperparameter Tuning
- **LinUCB α**: Start with 1.0, increase for more exploration
- **Neural Learning Rate**: 0.001-0.01 typical range
- **Neural Architecture**: Start with [64, 32] hidden layers
- **Epsilon**: 0.05-0.2 for ε-greedy exploration

### Model Selection Guide
| Scenario | Recommended Algorithm | Rationale |
|----------|----------------------|-----------|
| Linear rewards | LinUCB | Optimal for linear case, fast |
| Non-linear rewards | Neural Epsilon-Greedy | Captures complex patterns |
| High-dimensional features | Neural | Better scalability |
| Need fast inference | LinUCB | Closed-form predictions |
| Large action spaces | Neural | Better generalization |

## Code Examples

### Basic Usage
```python
from adapters.tf_agents.tf_agents_cb import TFAgentsCB

# Initialize LinUCB agent
model = TFAgentsCB(
    ctx_dim=10,      # Global context dimension
    act_dim=5,       # Per-arm feature dimension  
    num_actions=3,   # Number of available actions
    agent_type='lin_ucb',
    alpha=1.0        # Exploration parameter
)

# Training
context = np.array([0.1, 0.2, ...])  # Shape: (ctx_dim,)
action_features = np.array([1.0, 0.5, ...])  # Shape: (act_dim,)
reward = 0.8
model.update(context, action_features, reward)

# Prediction
actions = [action1_features, action2_features, action3_features]
chosen_action = model.choose(context, actions)
```

### Advanced Configuration
```python
# Neural agent with custom architecture
model = TFAgentsCB(
    ctx_dim=50,
    act_dim=20, 
    num_actions=10,
    agent_type='neural_epsilon_greedy',
    learning_rate=0.001
)

# Batch training
interactions = [
    (context1, action1, reward1, prob1),
    (context2, action2, reward2, prob2),
    # ...
]
model.batch_update(interactions)
```

## Performance Characteristics

| Algorithm | Training Speed | Inference Speed | Memory Usage | Scalability |
|-----------|---------------|----------------|--------------|-------------|
| LinUCB | Fast | Very Fast | Low | Good |
| Neural | Moderate | Fast | Moderate | Excellent |

## Integration with Your Library

### Follows Same Interface
- ✅ Implements `CBModel` protocol
- ✅ Same method signatures as `lin_reg_iwr.py`
- ✅ Compatible with existing registry system
- ✅ Same training script pattern

### Registry Integration
```python
# Save model
model.save(name="my_tf_model", version="v1", registry_root="./artifacts")

# Load model  
loaded_model = TFAgentsCB(ctx_dim=2, act_dim=2, num_actions=3)
loaded_model.load(name="my_tf_model", version="v1")
```

## Deployment Considerations

### Production Serving
- **TensorFlow Serving**: Export SavedModel format for high-throughput serving
- **Batch Prediction**: Process multiple contexts simultaneously
- **Model Versioning**: Use TF-Agents' built-in checkpoint system

### Monitoring
- **Exploration Rate**: Track how often non-greedy actions are chosen
- **Reward Distribution**: Monitor reward statistics over time
- **Model Performance**: Compare online vs offline evaluation metrics

### A/B Testing
- **Gradual Rollout**: Start with small traffic percentage
- **Control Group**: Compare against existing system
- **Safety Checks**: Implement minimum exploration rate

## Troubleshooting

### Common Issues
1. **OOM Errors**: Reduce batch size or network size
2. **Slow Convergence**: Increase learning rate or add more training data
3. **Poor Exploration**: Increase α (LinUCB) or ε (Neural)
4. **Overfitting**: Add regularization or reduce network complexity

### Debug Commands
```python
# Check model state
print(f"Initialized: {model.initialized}")
print(f"Agent type: {model.agent_type}")

# Inspect predictions
predictions = model.predict(context, actions)
print(f"Action scores: {predictions}")
```

## Research Papers

1. **LinUCB**: "A Contextual-Bandit Approach to Personalized News Article Recommendation" (Li et al., 2010)
2. **Neural Bandits**: "Deep Bayesian Bandits Showdown" (Riquelme et al., 2018) 
3. **TF-Agents**: "TF-Agents: A Library for Reinforcement Learning in TensorFlow" (Guadarrama et al., 2018)

## Next Steps

1. **Install Dependencies**: `pip install tensorflow tf-agents`
2. **Run Example**: `python scripts/train_tf_agents_cb.py`
3. **Integrate**: Replace existing models with TF-Agents versions
4. **Scale**: Deploy to production with TensorFlow Serving