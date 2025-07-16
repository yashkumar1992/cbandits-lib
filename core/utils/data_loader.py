# core/utils/data_loader.py
import json
from typing import List, Tuple, Any, Dict

class SimpleAction:
    """Simple action implementation for loading data"""
    def __init__(self, action_id: int, features: Dict[str, Any]):
        self.action_id = action_id
        self.action_features = features
    
    def get_id(self):
        return self.action_id
    
    def features(self):
        return self.action_features

def load_dat(file_path: str) -> List[Tuple[Any, List[SimpleAction]]]:
    """
    Load data from JSON lines file and return as list of (context, actions) tuples
    """
    rows = []
    
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            
            context = data['context']
            actions_data = data['actions']
            
            # Convert to SimpleAction objects
            actions = [
                SimpleAction(i, action_data) 
                for i, action_data in enumerate(actions_data)
            ]
            
            rows.append((context, actions))
    
    return rows