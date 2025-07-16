# core/utils/vw_parser.py
import re
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np

class VWFormatParser:
    """Parser for Vowpal Wabbit ADF format data"""
    
    def __init__(self):
        self.shared_line_pattern = re.compile(r'shared\s+\|(\w+)\s+(.*)')
        self.action_line_pattern = re.compile(r'(?:(\d+):([-+]?\d*\.?\d+):([-+]?\d*\.?\d+)\s+)?\|(\w+)\s+(.*)')
        
    def parse_features(self, feature_string: str) -> Dict[str, Union[float, str]]:
        """Parse feature string into dict of feature_name: value"""
        features = {}
        if not feature_string.strip():
            return features
            
        for feature in feature_string.split():
            if '=' in feature:
                key, value = feature.split('=', 1)
                # Try to convert to float, fallback to string
                try:
                    features[key] = float(value)
                except ValueError:
                    features[key] = value
            elif ':' in feature:
                key, value = feature.split(':', 1)
                try:
                    features[key] = float(value)
                except ValueError:
                    features[key] = value
            else:
                # Binary feature (just key, value=1)
                features[feature] = 1.0
        return features
    
    def parse_vw_block(self, lines: List[str]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Optional[Tuple[int, float, float]]]:
        """
        Parse a VW ADF block into context, actions, and optional label info
        
        Returns:
            context: Dict of context features
            actions: List of action feature dicts
            label_info: Optional (chosen_action_idx, reward, probability)
        """
        context = {}
        actions = []
        label_info = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Parse shared (context) line
            shared_match = self.shared_line_pattern.match(line)
            if shared_match:
                namespace = shared_match.group(1)
                feature_string = shared_match.group(2)
                context.update(self.parse_features(feature_string))
                continue
            
            # Parse action line
            action_match = self.action_line_pattern.match(line)
            if action_match:
                action_idx = action_match.group(1)
                cost = action_match.group(2)
                prob = action_match.group(3)
                namespace = action_match.group(4)
                feature_string = action_match.group(5)
                
                # Parse action features
                action_features = self.parse_features(feature_string)
                actions.append(action_features)
                
                # If this line has a label (cost/prob), record it
                if action_idx is not None and cost is not None and prob is not None:
                    reward = -float(cost)  # VW uses cost, we use reward
                    probability = float(prob)
                    chosen_idx = len(actions) - 1  # This action was chosen
                    label_info = (chosen_idx, reward, probability)
        
        return context, actions, label_info
    
    def parse_vw_file(self, file_path: str) -> List[Tuple[Dict[str, Any], List[Dict[str, Any]], Optional[Tuple[int, float, float]]]]:
        """
        Parse entire VW file into list of (context, actions, label_info) tuples
        """
        interactions = []
        current_block = []
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    # Empty line indicates end of block
                    if current_block:
                        context, actions, label_info = self.parse_vw_block(current_block)
                        interactions.append((context, actions, label_info))
                        current_block = []
                else:
                    current_block.append(line)
            
            # Handle last block if file doesn't end with empty line
            if current_block:
                context, actions, label_info = self.parse_vw_block(current_block)
                interactions.append((context, actions, label_info))
        
        return interactions
    
    def convert_to_arrays(self, interactions: List[Tuple[Dict[str, Any], List[Dict[str, Any]], Optional[Tuple[int, float, float]]]]) -> Tuple[np.ndarray, List[List[np.ndarray]], List[Optional[Tuple[int, float, float]]]]:
        """
        Convert parsed interactions to numpy arrays for ML algorithms
        
        Returns:
            contexts: numpy array of context vectors
            actions_list: list of action arrays for each interaction
            labels: list of label info tuples
        """
        if not interactions:
            return np.array([]), [], []
        
        # Extract all unique feature names to create consistent feature vectors
        all_context_features = set()
        all_action_features = set()
        
        for context, actions, _ in interactions:
            all_context_features.update(context.keys())
            for action in actions:
                all_action_features.update(action.keys())
        
        context_feature_names = sorted(list(all_context_features))
        action_feature_names = sorted(list(all_action_features))
        
        # Convert to arrays
        context_arrays = []
        actions_arrays = []
        labels = []
        
        for context, actions, label_info in interactions:
            # Convert context to vector
            context_vec = np.array([
                context.get(feat, 0.0) if isinstance(context.get(feat, 0.0), (int, float)) else hash(str(context.get(feat, 0.0))) % 1000 
                for feat in context_feature_names
            ], dtype=float)
            context_arrays.append(context_vec)
            
            # Convert actions to vectors
            action_vecs = []
            for action in actions:
                action_vec = np.array([
                    action.get(feat, 0.0) if isinstance(action.get(feat, 0.0), (int, float)) else hash(str(action.get(feat, 0.0))) % 1000
                    for feat in action_feature_names
                ], dtype=float)
                action_vecs.append(action_vec)
            actions_arrays.append(action_vecs)
            labels.append(label_info)
        
        return np.array(context_arrays), actions_arrays, labels

# Example usage and testing
if __name__ == "__main__":
    # Test with sample VW data
    sample_vw = """shared |C dayofweek=1
|A eventType=views timeWindow=1440 threshold=Minimum: 10
1:1.0:0.33 |A eventType=atc timeWindow=480 threshold=10-20
|A eventType=purchases timeWindow=60 threshold=20-30

shared |C dayofweek=2
0:1.0:0.33 |A eventType=atc timeWindow=480 threshold=10-20
|A eventType=views timeWindow=1440 threshold=Minimum: 10
|A eventType=purchases timeWindow=60 threshold=20-30"""
    
    # Save to temp file for testing
    with open('/tmp/test_vw.dat', 'w') as f:
        f.write(sample_vw)
    
    parser = VWFormatParser()
    interactions = parser.parse_vw_file('/tmp/test_vw.dat')
    
    print("Parsed interactions:")
    for i, (context, actions, label_info) in enumerate(interactions):
        print(f"\nInteraction {i+1}:")
        print(f"  Context: {context}")
        print(f"  Actions: {actions}")
        print(f"  Label: {label_info}")
    
    # Convert to arrays
    contexts, actions_arrays, labels = parser.convert_to_arrays(interactions)
    print(f"\nContext array shape: {contexts.shape}")
    print(f"Number of action sets: {len(actions_arrays)}")
    print(f"First context: {contexts[0]}")
    print(f"First action set shapes: {[a.shape for a in actions_arrays[0]]}")