import numpy as np
from typing import List, Tuple
from pearl.pearl_agent import PearlAgent
from pearl.policy_learners.contextual_bandits.neural_bandit import NeuralBandit
from pearl.policy_learners.contextual_bandits.neural_linear_bandit import (
    NeuralLinearBandit,
)
from pearl.policy_learners.exploration_modules.contextual_bandits.squarecb_exploration import SquareCBExploration
from pearl.action_representation_modules.one_hot_action_representation_module import OneHotActionTensorRepresentationModule
from pearl.replay_buffers import BasicReplayBuffer
from pearl.api.action_result import ActionResult

from registry.registry_fs import FSModelStore
from core.interfaces.cb_model import CBModel


class NeuralSquareCB(CBModel):
    def __init__(self, ctx_dim: int, act_dim: int, hidden_dims: List[int], gamma: float, learning_rate: float = 0.01):
        self.ctx_dim = ctx_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.feature_dim = ctx_dim + act_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate

        self.action_representation_module = OneHotActionTensorRepresentationModule(max_number_actions=act_dim)

        self.policy_learner = NeuralBandit(
            feature_dim=self.feature_dim,
            hidden_dims=hidden_dims,
            training_rounds=10,
            learning_rate=learning_rate,
            action_representation_module=self.action_representation_module,
            exploration_module=SquareCBExploration(gamma=gamma),
        )

        self.replay_buffer = BasicReplayBuffer(capacity=100_000)

        self.agent = PearlAgent(
            policy_learner=self.policy_learner,
            replay_buffer=self.replay_buffer,
            device_id=-1,
        )

    def predict(self, context: np.ndarray, actions: List[np.ndarray], eval_mode: bool = False):
        return self.agent.policy_learner.predict(context, actions)

    def choose(self, context: np.ndarray, actions: List[np.ndarray]):
        return self.agent.act(observation=context, available_actions=actions)

    def update(self, context: np.ndarray, action: np.ndarray, reward: float, chosen_prob: float = 1.0):    
        # Register latest action to internal state
        self.agent.act(observation=context, available_actions=[action])

        result = ActionResult(
            observation=context,
            reward=reward,
            terminated=False,
            truncated=False,
            info={"chosen_action": action}  # we pass the action here
        )
        self.agent.observe(result)
        self.agent.learn()
        return {"status": "updated"}, True


    def batch_update(self, interactions: List[Tuple[np.ndarray, np.ndarray, float, float]]):
        for context, chosen_action, reward, chosen_prob in interactions:
            self.update(context, chosen_action, reward, chosen_prob)

    def save(self, registry_root="./artifacts/models", name="neural_squarecb", version="1.0", artifact_subpath="models"):
        model_store = FSModelStore()
        model_store.save_torch(
            name=name,
            model_data=self.agent.policy_learner.model.state_dict(),
            version=version,
            registry_root=registry_root,
            artifact_subpath=artifact_subpath,
        )

    def load(self, registry_root="./artifacts/models", name="neural_squarecb", version="1.0", artifact_subpath="models"):
        model_store = FSModelStore()
        state_dict = model_store.load_torch(
            name=name,
            version=version,
            registry_root=registry_root,
            artifact_subpath=artifact_subpath,
        )
        self.agent.policy_learner.model.load_state_dict(state_dict)