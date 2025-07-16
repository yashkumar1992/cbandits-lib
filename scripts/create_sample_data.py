# scripts/create_sample_data.py
import json
import numpy as np

def create_sample_data():
    """Create sample contextual bandit data"""
    data = []
    
    for i in range(1000):
        # Sample context
        context = {
            "user_age": np.random.randint(18, 65),
            "time_of_day": np.random.randint(0, 24),
            "device_type": np.random.choice(["mobile", "desktop", "tablet"])
        }
        
        # Sample actions (e.g., different content recommendations)
        actions = [
            {"content_type": "news", "category": "sports", "length": "short"},
            {"content_type": "video", "category": "entertainment", "length": "medium"},
            {"content_type": "article", "category": "technology", "length": "long"}
        ]
        
        # Simulate choosing action and getting reward
        chosen_action_idx = np.random.randint(0, len(actions))
        reward = np.random.beta(2, 5)  # Simulate realistic reward distribution
        
        interaction = {
            "context": context,
            "actions": actions,
            "chosen_action": chosen_action_idx,
            "reward": reward,
            "probability": 1.0 / len(actions)  # uniform random policy
        }
        
        data.append(interaction)
    
    # Save as JSON lines
    with open('datasets/sample_interactions.jsonl', 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    create_sample_data()
    print("Sample data created in datasets/sample_interactions.jsonl")