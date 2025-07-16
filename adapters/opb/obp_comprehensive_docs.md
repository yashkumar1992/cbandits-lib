# Open Bandit Pipeline (OBP) Integration Guide

## Overview

This guide covers integrating Open Bandit Pipeline algorithms into your contextual bandits framework with support for VW format data and comprehensive algorithm configurations.

## Supported Algorithms

### 1. Linear Upper Confidence Bound (LinUCB)
**Best for**: High-dimensional contextual features, linear reward relationships
```yaml
open_bandits_adapter:
  algorithm: linear_ucb
  params:
    random_state: 12345
    # Note: dim and n_actions are auto-determined from data
```

**Mathematical Foundation**:
- Uses linear regression with confidence bounds
- Action selection: `a* = argmax(θᵀx + α√(xᵀA⁻¹x))`
- Balances exploitation vs exploration via confidence intervals

### 2. Logistic Upper Confidence Bound (LogisticUCB) 
**Best for**: Binary/categorical outcomes, non-linear reward relationships
```yaml
open_bandits_adapter:
  algorithm: logistic_ucb
  params:
    random_state: 12345
```

**Mathematical Foundation**:
- Uses logistic regression for probability estimation
- Better for bounded rewards in [0,1]
- Handles non-linear feature interactions

### 3. Linear Thompson Sampling (LinTS)
**Best for**: Bayesian uncertainty, fast exploration
```yaml
open_bandits_adapter:
  algorithm: linear_ts
  params:
    random_state: 12345
```

**Mathematical Foundation**:
- Bayesian approach with posterior sampling
- Samples from posterior distribution of parameters
- Often faster convergence than UCB methods

### 4. Logistic Thompson Sampling (LogisticTS)
**Best for**: Binary outcomes with Bayesian uncertainty
```yaml
open_bandits_adapter:
  algorithm: logistic_ts
  params:
    random_state: 12345
```

### 5. Linear Epsilon-Greedy (LinEpsilonGreedy)
**Best for**: Simple exploration, interpretable behavior
```yaml
open_bandits_adapter:
  algorithm: linear_epsilon_greedy
  params:
    epsilon: 0.1        # Exploration rate (0.05-0.3 typical)
    random_state: 12345
```

**Mathematical Foundation**:
- Linear model with ε-greedy exploration
- With probability ε: choose random action
- With probability 1-ε: choose greedy action

### 6. Epsilon-Greedy (Non-contextual)
**Best for**: Simple multi-armed bandits, baseline comparisons
```yaml
open_bandits_adapter:
  algorithm: epsilon_greedy
  params:
    epsilon: 0.1
    random_state: 12345
```

### 7. Random Policy
**Best for**: Baseline/control, data collection
```yaml
open_bandits_adapter:
  algorithm: random
  params:
    random_state: 12345
```

## Parameter Tuning Guide

### Epsilon (ε) Parameter
- **Range**: 0.01 - 0.3
- **Low (0.01-0.05)**: Heavy exploitation, good for stable environments
- **Medium (0.1-0.15)**: Balanced exploration/exploitation 
- **High (0.2-0.3)**: Heavy exploration, good for dynamic environments

### Random State
- **Purpose**: Reproducible results
- **Recommendation**: Use consistent values across experiments

## VW Format Data Structure

### Basic Format
```
shared |C <context_features>
[<action_idx>:<cost>:<probability>] |A <action_features>
|A <action_features>
...

shared |C <context_features>
...
```

### Feature Encoding
- **Categorical**: `feature_name=category_value`
- **Numerical**: `feature_name:numerical_value` or `feature_name=numerical_value`
- **Binary**: `feature_name` (implicitly 1.0)

### Example Data
```
shared |C dayofweek=1 hour=14
|A eventType=views timeWindow=1440 threshold=10
1:1.0:0.33 |A eventType=atc timeWindow=480 threshold=20
|A eventType=purchases timeWindow=60 threshold=30

shared |C dayofweek=2 hour=10
0:0.8:0.33 |A eventType=views timeWindow=1440 threshold=10
|A eventType=atc timeWindow=480 threshold=20
|A eventType=purchases timeWindow=60 threshold=30
```

## Usage Examples

### 1. Training from VW File
```python
from adapters.opb.opb_adapter_vw import OpenBanditsAdapterVW

# Initialize adapter
adapter = OpenBanditsAdapterVW('conf/obp_linear_ucb_config.yaml')

# Train from VW format file
adapter.batch_update(data_file='datasets/vw_bandit_dataset.dat')

# Save trained model
model_path = adapter.save('my_model', 'v1', 'artifacts')
```

### 2. Making Predictions
```python
from adapters.opb.opb_adapter_vw import SimpleActionVW

# Define context and actions
context = {"dayofweek": 3, "hour": 15}
actions = [
    SimpleActionVW(0, {"eventType": "views", "timeWindow": 1440}),
    SimpleActionVW(1, {"eventType": "atc", "timeWindow": 480}),
    SimpleActionVW(2, {"eventType": "purchases", "timeWindow": 60})
]

# Get prediction
action_id, probability = adapter.predict((context, actions))
print(f"Chosen action: {action_id} with probability: {probability}")
```

### 3. Loading Saved Models
```python
# Load previously trained model
loaded_adapter = OpenBanditsAdapterVW.load(
    name='my_model',
    version='v1', 
    registry_root='artifacts'
)

# Use for predictions
action_id, prob = loaded_adapter.predict((context, actions))
```

## Algorithm Selection Guide

### Problem Characteristics vs Algorithm Choice

| Problem Type | Context Size | Recommended Algorithm | Rationale |
|-------------|-------------|---------------------|-----------|
| E-commerce recommendations | High (>50 features) | LinUCB | Handles high-dimensional contexts well |
| Ad placement | Medium (10-50) | LinTS | Fast exploration, good for real-time |
| Content optimization | Low (<10) | Linear Epsilon-Greedy | Simple, interpretable |
| A/B testing baseline | Any | Random | Control/baseline comparison |
| Binary outcomes | Any | LogisticUCB/LogisticTS | Better for 0/1 rewards |

### Performance Characteristics

| Algorithm | Exploration Speed | Computational Cost | Memory Usage | Theoretical Guarantees |
|-----------|------------------|-------------------|--------------|----------------------|
| LinUCB | Medium | Medium | Medium | Strong (sublinear regret) |
| LinTS | Fast | Medium | Medium | Strong (Bayesian) |
| LogisticUCB | Medium | High | Medium | Strong |
| LogisticTS | Fast | High | Medium | Strong |
| Linear ε-Greedy | Slow | Low | Low | Weak |
| ε-Greedy | Slow | Very Low | Very Low | Weak |
| Random | N/A | Very Low | Very Low | None |

## Advanced Configuration

### Multi-Experiment Setup
```yaml
# conf/experiment_configs.yaml
experiments:
  aggressive_exploration:
    algorithm: linear_epsilon_greedy
    params:
      epsilon: 0.25
      random_state: 12345
  
  conservative_exploitation:
    algorithm: linear_ucb
    params:
      random_state: 12345
  
  bayesian_approach:
    algorithm: linear_ts
    params:
      random_state: 12345
```

### Hyperparameter Sweeps
```python
# Example hyperparameter sweep
epsilons = [0.05, 0.1, 0.15, 0.2]
algorithms = ['linear_ucb', 'linear_ts', 'linear_epsilon_greedy']

results = {}
for algo in algorithms:
    for eps in epsilons:
        if 'epsilon' in algo:
            config = create_config(algo, epsilon=eps)
        else:
            config = create_config(algo)
        
        adapter = OpenBanditsAdapterVW(config)
        # ... train and evaluate
        results[(algo, eps)] = evaluate_performance(adapter)
```

## Industry Best Practices

### 1. Data Collection
- **Exploration Phase**: Start with ε-greedy (ε=0.2-0.3) for initial data
- **Exploitation Phase**: Switch to LinUCB/LinTS after sufficient exploration
- **Safety**: Always maintain minimum exploration rate (ε≥0.01)

### 2. Feature Engineering
- **Normalization**: Scale features to similar ranges
- **Interaction Terms**: Consider feature interactions for complex domains
- **Categorical Encoding**: Use consistent encoding schemes

### 3. Model Evaluation
- **Off-Policy Evaluation**: Use OBP's built-in OPE estimators
- **A/B Testing**: Compare against current production system
- **Regret Analysis**: Monitor cumulative regret over time

### 4. Production Deployment
- **Monitoring**: Track exploration rate, reward distribution
- **Fallbacks**: Always have a default action for edge cases
- **Updates**: Retrain models with fresh data regularly

## Troubleshooting

### Common Issues

1. **"Unknown algorithm" Error**
   - Check algorithm name spelling in config
   - Ensure algorithm is in ALGORITHMS registry

2. **Dimension Mismatch**
   - Verify feature consistency between training/prediction
   - Check context and action feature names

3. **Poor Performance**
   - Increase exploration rate (epsilon)
   - Check feature quality and relevance
   - Consider algorithm switch (LinTS for faster exploration)

4. **Memory Issues**
   - Use smaller batch sizes
   - Consider non-contextual algorithms for very high dimensions

### Debug Commands
```python
# Check parsed data structure
from core.utils.vw_parser import VWFormatParser
parser = VWFormatParser()
interactions = parser.parse_vw_file('datasets/vw_bandit_dataset.dat')
print(f"Parsed {len(interactions)} interactions")
print(f"Sample: {interactions[0]}")

# Check feature dimensions
contexts, actions_arrays, labels = parser.convert_to_arrays(interactions)
print(f"Context shape: {contexts.shape}")
print(f"Action shapes: {[len(actions) for actions in actions_arrays]}")
```

## Running the Complete Example

1. **Install Dependencies**:
```bash
pip install obp pyyaml joblib numpy
```

2. **Prepare Your VW Data**:
   - Place your VW format file in `datasets/vw_bandit_dataset.dat`
   - Or use the provided sample data generator

3. **Run Training Script**:
```bash
python scripts/train_obp_vw_format.py
```

4. **Test Different Algorithms**:
   - Script will test all available algorithms
   - Compare performance across different approaches
   - Models will be saved for future use

This comprehensive setup provides a production-ready contextual bandit system with industry-standard algorithms and best practices!