import yaml
from typing import Any, Sequence, Tuple, Optional
from core.interfaces.cb_model import CBModel
from core.interfaces.action import Action
from open_bandits import OpenBanditsPipeline  # replace with actual import

class OpenBanditsAdapter(CBModel):
    def __init__(self, config_path: str) -> None:
        """
        Adapter to wrap Open Bandits Pipeline algorithms into our CBModel interface.
        """
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        ob_cfg = cfg.get('open_bandits_adapter', {})
        algorithm = ob_cfg['algorithm']
        params = ob_cfg.get('params', {})

        # Instantiate the Open Bandits model
        self.pipeline = OpenBanditsPipeline(algorithm=algorithm, **params)

    def get_exploration_strategy(self) -> Any:
        return self.pipeline.get_exploration()

    def predict(
        self,
        row: Tuple[Any, Sequence[Action]],
        eval_mode: bool = False
    ) -> Tuple[Any, float]:
        context, actions = row
        ob_actions = [a.features() for a in actions]
        chosen_idx, score = self.pipeline.predict(context, ob_actions, eval=eval_mode)
        chosen_action = actions[chosen_idx]
        return chosen_action.get_id(), score

    def batch_predict(
        self,
        rows: Sequence[Tuple[Any, Sequence[Action]]],
        eval_mode: bool = False
    ) -> Sequence[Tuple[Any, float]]:
        return [self.predict(row, eval_mode) for row in rows]

    def update(
        self,
        interaction: Tuple[Any, Any, float, Optional[float]]
    ) -> None:
        context, action_id, reward, prob = interaction
        self.pipeline.learn(context, action_id, reward, prob)

    def batch_update(
        self,
        interactions: Optional[Sequence[Tuple[Any, Any, float, Optional[float]]]] = None,
        data_file: Optional[str] = None
    ) -> None:
        if interactions:
            for inter in interactions:
                self.update(inter)
        elif data_file:
            self.pipeline.learn_from_file(data_file)

    def reset(self) -> None:
        self.pipeline.reset()

    def save(
        self,
        name: str,
        version: str,
        registry_root: str
    ) -> str:
        return self.pipeline.save(name, version, registry_root)

    @classmethod
    def load(
        cls,
        name: str,
        version: str,
        registry_root: str
    ) -> CBModel:
        inst = cls.__new__(cls)
        inst.pipeline = OpenBanditsPipeline.load(name, version, registry_root)
        return inst