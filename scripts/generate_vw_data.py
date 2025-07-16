# scripts/generate_vw_data.py
import numpy as np
import random
from typing import List, Tuple, Dict

def generate_realistic_vw_data(n_samples: int = 1000, output_file: str = 'datasets/vw_bandit_dataset.dat'):
    """
    Generate realistic VW format contextual bandit data for e-commerce recommendation
    """
    
    # Context features (user and session characteristics)
    user_segments = ['new_user', 'returning_user', 'premium_user', 'casual_user']
    device_types = ['mobile', 'desktop', 'tablet']
    time_segments = ['morning', 'afternoon', 'evening', 'night']
    
    # Action features (recommendation types)
    event_types = ['views', 'atc', 'purchases']  # add-to-cart
    time_windows = [60, 480, 1440]  # 1 hour, 8 hours, 24 hours
    thresholds = ['low', 'medium', 'high']
    
    # Reward simulation logic
    def simulate_reward(context: Dict, action: Dict) -> float:
        """Simulate realistic reward based on context-action interaction"""
        base_reward = 0.3
        
        # User segment effects
        if context['user_segment'] == 'premium_user':
            base_reward += 0.2
        elif context['user_segment'] == 'new_user':
            base_reward += 0.1
        
        # Time of day effects
        if context['time_segment'] == 'evening':
            base_reward += 0.15
        elif context['time_segment'] == 'afternoon':
            base_reward += 0.1
        
        # Device effects
        if context['device_type'] == 'mobile':
            base_reward += 0.1
        
        # Action-specific effects
        if action['eventType'] == 'purchases':
            base_reward += 0.3
        elif action['eventType'] == 'atc':
            base_reward += 0.15
        
        # Time window effects
        if action['timeWindow'] == 1440:  # 24 hours
            base_reward += 0.1
        
        # Threshold effects
        if action['threshold'] == 'high':
            base_reward += 0.1
        elif action['threshold'] == 'medium':
            base_reward += 0.05
        
        # Add noise
        noise = np.random.normal(0, 0.1)
        final_reward = max(0.0, min(1.0, base_reward + noise))
        
        return final_reward
    
    # Policy simulation (epsilon-greedy with domain knowledge)
    def select_action(context: Dict, actions: List[Dict], epsilon: float = 0.3) -> Tuple[int, float]:
        """Simulate policy that chooses actions with some domain knowledge"""
        
        if random.random() < epsilon:
            # Random exploration
            chosen_idx = random.randint(0, len(actions) - 1)
        else:
            # Greedy exploitation with domain heuristics
            scores = []
            for action in actions:
                score = 0.0
                
                # Premium users prefer purchase actions
                if context['user_segment'] == 'premium_user' and action['eventType'] == 'purchases':
                    score += 0.5
                
                # Mobile users prefer quick actions
                if context['device_type'] == 'mobile' and action['timeWindow'] <= 480:
                    score += 0.3
                
                # Evening time favors higher engagement
                if context['time_segment'] == 'evening':
                    score += 0.2
                
                # High thresholds generally better
                if action['threshold'] == 'high':
                    score += 0.2
                
                scores.append(score + random.random() * 0.1)  # Add noise
            
            chosen_idx = scores.index(max(scores))
        
        # Calculate probability (simplified uniform for exploration)
        probability = 1.0 / len(actions)
        
        return chosen_idx, probability
    
    # Generate data
    vw_lines = []
    
    for i in range(n_samples):
        # Sample context
        context = {
            'dayofweek': random.randint(1, 7),
            'hour': random.randint(0, 23),
            'user_segment': random.choice(user_segments),
            'device_type': random.choice(device_types),
            'time_segment': random.choice(time_segments),
            'session_length': round(random.uniform(1.0, 120.0), 1)  # minutes
        }
        
        # Sample actions (always 3 actions for consistency)
        actions = []
        for j in range(3):
            action = {
                'eventType': event_types[j],
                'timeWindow': time_windows[j],
                'threshold': random.choice(thresholds)
            }
            actions.append(action)
        
        # Select action using policy
        chosen_idx, probability = select_action(context, actions)
        
        # Calculate reward
        reward = simulate_reward(context, actions[chosen_idx])
        cost = -reward  # VW uses cost (negative reward)
        
        # Format as VW
        # Shared line (context)
        context_features = []
        for key, value in context.items():
            if isinstance(value, str):
                context_features.append(f"{key}={value}")
            else:
                context_features.append(f"{key}:{value}")
        
        shared_line = f"shared |C {' '.join(context_features)}"
        vw_lines.append(shared_line)
        
        # Action lines
        for j, action in enumerate(actions):
            action_features = []
            for key, value in action.items():
                if isinstance(value, str):
                    action_features.append(f"{key}={value}")
                else:
                    action_features.append(f"{key}:{value}")
            
            # Add label if this action was chosen
            if j == chosen_idx:
                action_line = f"{chosen_idx}:{cost:.3f}:{probability:.3f} |A {' '.join(action_features)}"
            else:
                action_line = f"|A {' '.join(action_features)}"
            
            vw_lines.append(action_line)
        
        # Add empty line between samples
        vw_lines.append("")
    
    # Write to file
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(vw_lines))
    
    print(f"✅ Generated {n_samples} VW interactions in {output_file}")
    
    # Print sample
    print("\n📄 Sample VW data:")
    print('\n'.join(vw_lines[:12]))  # Show first interaction
    
    return output_file

def create_test_cases():
    """Create specific test cases for evaluation"""
    test_cases = [
        # Premium user, evening, mobile -> should prefer purchases
        {
            'context': {'dayofweek': 5, 'hour': 20, 'user_segment': 'premium_user', 
                       'device_type': 'mobile', 'time_segment': 'evening', 'session_length': 45.0},
            'expected_best': 2  # purchases action
        },
        # New user, morning, desktop -> should prefer views
        {
            'context': {'dayofweek': 2, 'hour': 9, 'user_segment': 'new_user',
                       'device_type': 'desktop', 'time_segment': 'morning', 'session_length': 15.0},
            'expected_best': 0  # views action
        },
        # Casual user, afternoon, tablet -> should prefer atc
        {
            'context': {'dayofweek': 3, 'hour': 14, 'user_segment': 'casual_user',
                       'device_type': 'tablet', 'time_segment': 'afternoon', 'session_length': 30.0},
            'expected_best': 1  # atc action
        }
    ]
    
    # Create test VW format
    test_lines = []
    for i, test_case in enumerate(test_cases):
        context = test_case['context']
        
        # Standard actions
        actions = [
            {'eventType': 'views', 'timeWindow': 1440, 'threshold': 'medium'},
            {'eventType': 'atc', 'timeWindow': 480, 'threshold': 'medium'},
            {'eventType': 'purchases', 'timeWindow': 60, 'threshold': 'high'}
        ]
        
        # Format context
        context_features = []
        for key, value in context.items():
            if isinstance(value, str):
                context_features.append(f"{key}={value}")
            else:
                context_features.append(f"{key}:{value}")
        
        shared_line = f"shared |C {' '.join(context_features)}"
        test_lines.append(shared_line)
        
        # Format actions (no labels for prediction)
        for action in actions:
            action_features = []
            for key, value in action.items():
                action_features.append(f"{key}={value}")
            action_line = f"|A {' '.join(action_features)}"
            test_lines.append(action_line)
        
        test_lines.append("")  # Empty line between test cases
    
    # Save test cases
    test_file = 'datasets/vw_test_cases.dat'
    with open(test_file, 'w') as f:
        f.write('\n'.join(test_lines))
    
    print(f"✅ Created test cases in {test_file}")
    return test_file, test_cases

if __name__ == "__main__":
    # Generate training data
    train_file = generate_realistic_vw_data(n_samples=1000)
    
    # Generate test cases
    test_file, test_cases = create_test_cases()
    
    print(f"\n🎯 Files created:")
    print(f"  📈 Training data: {train_file}")
    print(f"  🧪 Test cases: {test_file}")
    print(f"\n💡 Test expectations:")
    for i, case in enumerate(test_cases):
        print(f"  Test {i+1}: {case['context']['user_segment']} user should prefer action {case['expected_best']}")