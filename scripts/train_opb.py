#!/usr/bin/env python3
# scripts/train_opb_complete.py

import os
import sys
import json
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adapters.opb.opb_adapter import OpenBanditsAdapter

def create_sample_config():
    """Create a sample config file"""
    config = {
        'open_bandits_adapter': {
            'algorithm': 'linear_ucb',
            'params': {
                'alpha': 1.0,
                'lambda_': 0.5
            }
        }
    }
    
    os.makedirs('conf', exist_ok=True)
    with open('conf/opb_config.yaml', 'w') as f:
        import yaml
        yaml.dump(config, f)
    
    return 'conf/opb_config.yaml'

def create_sample_data():
    """Create sample contextual bandit data"""
    os.makedirs('datasets', exist_ok=True)
    
    data = []
    for i in range(100):
        # Sample context
        context = {
            "user_age": float(np.random.randint(18, 65)),
            "time_of_day": float(np.random.randint(0, 24)),
        }
        
        # Sample actions
        actions = [
            {"content_type": 1.0, "category": 1.0},  # sports
            {"content_type": 2.0, "category": 2.0},  # entertainment  
            {"content_type": 3.0, "category": 3.0},  # technology
        ]
        
        # Simulate reward based on some logic
        chosen_action_idx = np.random.randint(0, len(actions))
        if context["user_age"] < 30 and chosen_action_idx == 1:  # young users like entertainment
            reward = np.random.beta(3, 2)
        elif context["user_age"] >= 50 and chosen_action_idx == 2:  # older users like tech
            reward = np.random.beta(3, 2)
        else:
            reward = np.random.beta(1, 3)  # lower reward otherwise
        
        interaction = {
            "context": context,
            "actions": actions,
            "chosen_action": chosen_action_idx,
            "reward": reward,
            "probability": 1.0 / len(actions)
        }
        
        data.append(interaction)
    
    # Save as JSON lines
    with open('datasets/sample_interactions.jsonl', 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    
    return 'datasets/sample_interactions.jsonl'

def main():
    print("🚀 Starting OBP Training Demo")
    
    # Create sample files
    config_path = create_sample_config()
    data_path = create_sample_data()
    
    print(f"✅ Created config: {config_path}")
    print(f"✅ Created data: {data_path}")
    
    # Initialize adapter
    print("🔧 Initializing OpenBandits adapter...")
    adapter = OpenBanditsAdapter(config_path)
    
    # Train
    print("🎯 Training model...")
    adapter.batch_update(data_file=data_path)
    
    # Save model
    print("💾 Saving model...")
    os.makedirs('artifacts', exist_ok=True)
    version = adapter.save(name='opb_model', version='v1', registry_root='artifacts')
    print(f"✅ Model saved at: {version}")
    
    # Demo predictions
    print("🔮 Running demo predictions...")
    
    # Load some test data
    test_contexts = [
        {"user_age": 25.0, "time_of_day": 14.0},
        {"user_age": 55.0, "time_of_day": 9.0},
        {"user_age": 35.0, "time_of_day": 20.0}
    ]
    
    test_actions = [
        {"content_type": 1.0, "category": 1.0},  # sports
        {"content_type": 2.0, "category": 2.0},  # entertainment
        {"content_type": 3.0, "category": 3.0},  # technology
    ]
    
    # Convert to action objects
    from core.utils.data_loader import SimpleAction
    action_objects = [SimpleAction(i, action) for i, action in enumerate(test_actions)]
    
    for i, context in enumerate(test_contexts):
        action_id, score = adapter.predict((context, action_objects))
        print(f"Test {i+1}: Context={context}")
        print(f"  → Chosen action: {action_id} (score={score:.4f})")
        print(f"  → Action details: {test_actions[action_id]}")
    
    print("🎉 Demo completed successfully!")

if __name__ == '__main__':
    main()