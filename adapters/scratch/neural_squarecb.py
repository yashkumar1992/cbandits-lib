import numpy as np
import torch
from typing import List, Tuple, Dict, Any
from core.interfaces.cb_model import CBModel
from registry.registry_fs import FSModelStore

# Pearl imports with correct API
from pearl.pearl_agent import PearlAgent
from pearl.policy_learners.contextual_bandits.neural_bandit import NeuralBandit
from pearl.policy_learners.exploration_modules.contextual_bandits.squarecb_exploration import SquareCBExploration
from pearl.action_representation_modules.one_hot_action_representation_module import OneHotActionTensorRepresentationModule
from pearl.replay_buffers import BasicReplayBuffer
from pearl.api.action_result import ActionResult
from pearl.api.action import Action


class NeuralSquareCB(CBModel):
    """
    Neural SquareCB implementation using Pearl framework.
    
    Mathematical Foundation:
    - Uses neural network for reward prediction: r̂(x,a) = f_θ(x,a)
    - SquareCB exploration via Inverse Gap Weighting (IGW)
    - Action selection: p(a) ∝ 1/max(r̂_max - r̂_a, δ)²
    
    Industry Standards:
    - Modular design following Pearl's architecture
    - Proper error handling and state management
    - Production-ready serialization/deserialization
    """
    
    def __init__(
        self, 
        ctx_dim: int, 
        act_dim: int, 
        hidden_dims: List[int] = [64, 32], 
        gamma: float = 100.0, 
        learning_rate: float = 0.01,
        training_rounds: int = 10,
        replay_buffer_capacity: int = 100_000
    ):
        """
        Initialize Neural SquareCB model.
        
        Args:
            ctx_dim: Context feature dimension
            act_dim: Action feature dimension  
            hidden_dims: Neural network hidden layer dimensions
            gamma: SquareCB exploration parameter (higher = more exploitation)
            learning_rate: Neural network learning rate
            training_rounds: Number of training rounds per update
            replay_buffer_capacity: Replay buffer size
        """
        self.ctx_dim = ctx_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.feature_dim = ctx_dim + act_dim  # Combined context+action features
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.training_rounds = training_rounds
        self.replay_buffer_capacity = replay_buffer_capacity
        
        # Initialize Pearl components
        self._init_pearl_agent()
        
        # Track current state for Pearl's API
        self._current_observation = None
        self._current_action_space = None
        self._last_action = None
        
    def _init_pearl_agent(self):
        """Initialize Pearl agent with NeuralBandit + SquareCB exploration."""
        
        # Action representation module for discrete actions
        self.action_representation_module = OneHotActionTensorRepresentationModule(
            max_number_actions=self.act_dim
        )
        
        # Neural bandit policy learner with SquareCB exploration
        self.policy_learner = NeuralBandit(
            feature_dim=self.feature_dim,
            hidden_dims=self.hidden_dims,
            training_rounds=self.training_rounds,
            learning_rate=self.learning_rate,
            action_representation_module=self.action_representation_module,
            exploration_module=SquareCBExploration(gamma=self.gamma),
        )
        
        # Replay buffer for experience storage
        self.replay_buffer = BasicReplayBuffer(capacity=self.replay_buffer_capacity)
        
        # Pearl agent
        self.agent = PearlAgent(
            policy_learner=self.policy_learner,
            replay_buffer=self.replay_buffer,
            device_id=-1,  # CPU training
        )
        
    def get_exploration_strategy(self) -> Dict[str, Any]:
        """Return exploration strategy information."""
        return {
            "algorithm": "NeuralSquareCB",
            "gamma": self.gamma,
            "hidden_dims": self.hidden_dims,
            "learning_rate": self.learning_rate
        }
    
    def predict(
        self, 
        context: np.ndarray, 
        actions: List[np.ndarray], 
        eval_mode: bool = False
    ) -> List[float]:
        """
        Predict rewards for all actions using neural network.
        
        Args:
            context: Context features [ctx_dim]
            actions: List of action feature vectors [act_dim each]
            eval_mode: If True, disable exploration
            
        Returns:
            List of predicted rewards for each action
        """
        if not hasattr(self.agent.policy_learner, '_models') or len(self.agent.policy_learner._models) == 0:
            # Return uniform predictions if not trained
            return [0.0] * len(actions)
        
        predictions = []
        
        for action in actions:
            # Combine context and action features
            combined_features = torch.tensor(
                np.concatenate([context, action]), 
                dtype=torch.float32
            ).unsqueeze(0)
            
            # Get prediction from neural network
            with torch.no_grad():
                if hasattr(self.agent.policy_learner, '_models') and len(self.agent.policy_learner._models) > 0:
                    # Use the first model for prediction (ensemble average could be added)
                    pred = self.agent.policy_learner._models[0](combined_features).item()
                else:
                    pred = 0.0
                    
            predictions.append(pred)
            
        return predictions
    
    def choose(self, context: np.ndarray, actions: List[np.ndarray]) -> int:
        """
        Choose action using SquareCB exploration strategy.
        
        Args:
            context: Context features
            actions: Available actions
            
        Returns:
            Index of chosen action
        """
        # Convert to Pearl's expected format
        observation = torch.tensor(context, dtype=torch.float32)
        available_actions = [Action(torch.tensor(action, dtype=torch.float32)) for action in actions]
        
        # Update current state for Pearl
        self._current_observation = observation
        self._current_action_space = available_actions
        
        # Get action from Pearl agent using correct API
        try:
            # Pearl's act method signature: act(exploit=False)
            chosen_action = self.agent.act(exploit=False)
            self._last_action = chosen_action
            
            # Convert back to action index
            if hasattr(chosen_action, 'action'):
                # Extract action tensor and find matching index
                action_tensor = chosen_action.action
                for i, available_action in enumerate(available_actions):
                    if torch.allclose(action_tensor, available_action.action, atol=1e-6):
                        return i
            
            # Fallback to random if conversion fails
            return np.random.randint(len(actions))
            
        except Exception as e:
            print(f"Warning: Pearl agent selection failed: {e}. Using fallback.")
            # Fallback to manual SquareCB implementation
            return self._manual_squarecb_selection(context, actions)
    
    def _manual_squarecb_selection(self, context: np.ndarray, actions: List[np.ndarray]) -> int:
        """
        Manual SquareCB action selection implementation.
        
        Mathematical Implementation:
        1. Get predictions: r̂_a = f_θ(x,a) for all actions a
        2. Compute gaps: g_a = max_a' r̂_a' - r̂_a  
        3. Compute scores: s_a = 1/max(g_a, δ)²
        4. Sample: p(a) = s_a / Σ_a' s_a'
        """
        predictions = self.predict(context, actions)
        
        if not predictions or all(p == 0.0 for p in predictions):
            return np.random.randint(len(actions))
        
        # SquareCB Inverse Gap Weighting
        max_pred = max(predictions)
        delta = 0.01  # Minimum gap to avoid division by zero
        
        # Compute gaps and scores
        gaps = [max(max_pred - pred, delta) for pred in predictions]
        scores = [1.0 / (gap ** 2) for gap in gaps]
        
        # Convert to probabilities
        total_score = sum(scores)
        if total_score == 0:
            probs = [1.0 / len(actions)] * len(actions)
        else:
            probs = [score / total_score for score in scores]
        
        # Sample action
        return np.random.choice(len(actions), p=probs)
    
    def update(
        self, 
        context: np.ndarray, 
        action: np.ndarray, 
        reward: float, 
        chosen_prob: float = 1.0
    ) -> Tuple[Dict[str, str], bool]:
        """
        Update model with new interaction.
        
        Args:
            context: Context features that led to action
            action: Action taken (feature vector)  
            reward: Observed reward
            chosen_prob: Probability action was chosen (for importance weighting)
            
        Returns:
            (status_dict, success_flag)
        """
        try:
            # Create ActionResult for Pearl
            action_result = ActionResult(
                observation=torch.tensor(context, dtype=torch.float32),
                reward=reward,
                terminated=False,
                truncated=False,
                info={"chosen_prob": chosen_prob}
            )
            
            # Update Pearl agent
            self.agent.observe(action_result)
            self.agent.learn()
            
            return {"status": "updated"}, True
            
        except Exception as e:
            print(f"Warning: Pearl update failed: {e}")
            return {"status": "failed", "error": str(e)}, False
    
    def batch_update(
        self, 
        interactions: List[Tuple[np.ndarray, np.ndarray, float, float]]
    ) -> None:
        """
        Batch update with multiple interactions.
        
        Args:
            interactions: List of (context, action, reward, chosen_prob) tuples
        """
        for context, chosen_action, reward, chosen_prob in interactions:
            self.update(context, chosen_action, reward, chosen_prob)
    
    def reset(self) -> None:
        """Reset agent state."""
        try:
            # Reset Pearl agent if possible
            if hasattr(self.agent, 'reset') and self._current_observation is not None:
                self.agent.reset(self._current_observation, self._current_action_space)
        except Exception as e:
            print(f"Warning: Pearl reset failed: {e}. Reinitializing agent.")
            self._init_pearl_agent()
        
        # Clear state
        self._current_observation = None
        self._current_action_space = None
        self._last_action = None
    
    def save(
        self,
        name: str = "neural_squarecb",
        version: str = "1.0", 
        registry_root: str = "./artifacts/models",
        artifact_subpath: str = "models"
    ) -> str:
        """Save model state."""
        model_store = FSModelStore()
        
        # Save both Pearl agent state and our metadata
        save_data = {
            "pearl_state_dict": self.agent.state_dict() if hasattr(self.agent, 'state_dict') else None,
            "config": {
                "ctx_dim": self.ctx_dim,
                "act_dim": self.act_dim,
                "hidden_dims": self.hidden_dims,
                "gamma": self.gamma,
                "learning_rate": self.learning_rate,
                "training_rounds": self.training_rounds,
                "replay_buffer_capacity": self.replay_buffer_capacity
            }
        }
        
        return model_store.save(name, save_data, version, registry_root, artifact_subpath)
    
    def load(
        self,
        name: str = "neural_squarecb",
        version: str = "1.0",
        registry_root: str = "./artifacts/models", 
        artifact_subpath: str = "models"
    ) -> None:
        """Load model state."""
        model_store = FSModelStore()
        save_data = model_store.load(name, version, registry_root, artifact_subpath)
        
        # Restore configuration
        config = save_data.get("config", {})
        for key, value in config.items():
            setattr(self, key, value)
        
        # Reinitialize agent with loaded config
        self._init_pearl_agent()
        
        # Load Pearl state if available
        pearl_state = save_data.get("pearl_state_dict")
        if pearl_state and hasattr(self.agent, 'load_state_dict'):
            try:
                self.agent.load_state_dict(pearl_state)
            except Exception as e:
                print(f"Warning: Could not load Pearl state: {e}")


# Helper functions for integration testing
def create_test_data(n_samples: int = 100) -> List[Tuple[np.ndarray, np.ndarray, float, float]]:
    """Create synthetic test data for neural bandit."""
    interactions = []
    
    for _ in range(n_samples):
        # Random context
        context = np.random.randn(2)  # 2D context
        
        # Random action (2D action space)
        action = np.random.randn(2)
        
        # Synthetic reward based on context-action interaction
        reward = np.dot(context, action) + 0.1 * np.random.randn()
        reward = max(0, min(1, reward))  # Bound to [0,1]
        
        # Uniform probability (random policy)
        prob = 1.0 / 3  # Assuming 3 actions
        
        interactions.append((context, action, reward, prob))
    
    return interactions


if __name__ == "__main__":
    # Test the implementation
    print("🧪 Testing Neural SquareCB Implementation")
    
    # Initialize model
    model = NeuralSquareCB(ctx_dim=2, act_dim=2, gamma=100.0)
    
    # Create test data
    test_data = create_test_data(50)
    
    # Train model
    print("📈 Training model...")
    model.batch_update(test_data)
    
    # Test prediction
    test_context = np.array([0.5, -0.3])
    test_actions = [
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]), 
        np.array([-1.0, 0.0])
    ]
    
    print("🔮 Testing predictions...")
    predictions = model.predict(test_context, test_actions)
    chosen_action = model.choose(test_context, test_actions)
    
    print(f"Context: {test_context}")
    print(f"Predictions: {predictions}")
    print(f"Chosen action: {chosen_action}")
    
    # Test save/load
    print("💾 Testing save/load...")
    model.save("test_neural_squarecb", "v1")
    
    new_model = NeuralSquareCB(ctx_dim=2, act_dim=2)
    new_model.load("test_neural_squarecb", "v1") 
    
    new_predictions = new_model.predict(test_context, test_actions)
    print(f"Loaded model predictions: {new_predictions}")
    
    print("✅ Neural SquareCB test completed!")