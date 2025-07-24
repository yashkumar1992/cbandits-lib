import numpy as np
import tensorflow as tf

from tf_agents.trajectories import time_step as ts
from tf_agents.specs import tensor_spec
from tf_agents.bandits.agents import neural_epsilon_greedy_agent, neural_linucb_agent
from tf_agents.bandits.networks import global_and_arm_dot_product_network
from tf_agents.utils import common


class TFAgentsCB:
    def __init__(
        self,
        ctx_dim: int,
        act_dim: int,
        num_actions: int = None,
        agent_type: str = "neural_epsilon_greedy",
        learning_rate: float = 0.01,
        epsilon: float = 0.1,
        alpha: float = 1.0
    ):
        self.ctx_dim = ctx_dim
        self.act_dim = act_dim
        self.num_actions = num_actions
        self.agent_type = agent_type
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.alpha = alpha

        self._build_specs()
        self._setup_agent()

    def _build_specs(self):
        self.observation_spec = {
            "global": tf.TensorSpec([self.ctx_dim], dtype=tf.float32),
            "per_arm": tf.TensorSpec([None, self.act_dim], dtype=tf.float32)
        }
        self.time_step_spec = ts.time_step_spec(self.observation_spec)
        self.action_spec = tf.TensorSpec([], dtype=tf.int32)

    def _setup_agent(self):
        if self.agent_type == "neural_epsilon_greedy":
            self.encoding_network = global_and_arm_dot_product_network.GlobalAndArmDotProductNetwork(
                input_spec=self.observation_spec,
                global_layers=(16,),
                arm_layers=(8,),
                activation_fn=tf.nn.relu
            )

            self.agent = neural_epsilon_greedy_agent.NeuralEpsilonGreedyAgent(
                time_step_spec=self.time_step_spec,
                action_spec=self.action_spec,
                encoding_network=self.encoding_network,
                optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                epsilon=self.epsilon,
                emit_policy_info=(),
                accepts_per_arm_features=True
            )

        elif self.agent_type == "neural_lin_ucb":
            self.encoding_network = global_and_arm_dot_product_network.GlobalAndArmDotProductNetwork(
                input_spec=self.observation_spec,
                global_layers=(16,),
                arm_layers=(8,),
                activation_fn=tf.nn.relu
            )

            self.agent = neural_linucb_agent.NeuralLinUCBAgent(
                time_step_spec=self.time_step_spec,
                action_spec=self.action_spec,
                encoding_network=self.encoding_network,
                optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                alpha=self.alpha,
                emit_policy_info=(),
                accepts_per_arm_features=True
            )

        else:
            raise ValueError(f"Unknown agent type: {self.agent_type}")

        self.agent.initialize()

    def predict(self, context: np.ndarray, actions: list[np.ndarray]):
        context_tensor = tf.convert_to_tensor([context], dtype=tf.float32)
        arms_tensor = tf.convert_to_tensor([np.stack(actions)], dtype=tf.float32)

        observations = {
            "global": context_tensor,
            "per_arm": arms_tensor
        }

        scores = self.encoding_network(observations, training=False)
        return scores.numpy().flatten().tolist()

    def choose(self, context: np.ndarray, actions: list[np.ndarray]):
        scores = self.predict(context, actions)
        return int(np.argmax(scores))

    def update(self, context: np.ndarray, action: np.ndarray, reward: float, chosen_prob: float = 1.0):
        context_tensor = tf.convert_to_tensor([context], dtype=tf.float32)
        arms_tensor = tf.convert_to_tensor([[action]], dtype=tf.float32)

        timestep = ts.restart({
            "global": context_tensor[0],
            "per_arm": arms_tensor[0]
        }, batch_size=1)

        action_tensor = tf.convert_to_tensor([0], dtype=tf.int32)
        reward_tensor = tf.convert_to_tensor([reward], dtype=tf.float32)

        next_timestep = ts.transition({
            "global": context_tensor[0],
            "per_arm": arms_tensor[0]
        }, reward=reward_tensor)

        experience = tf.nest.pack_sequence_as(self.agent.collect_data_spec,
                                              [timestep, action_tensor, next_timestep])

        self.agent.train(experience)
        return {"status": "updated"}, True
