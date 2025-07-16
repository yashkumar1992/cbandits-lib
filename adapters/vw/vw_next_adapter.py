# adapters/vw/vw_next_adapter.py

from typing import Any, Sequence, Tuple, Optional, Dict
from vowpal_wabbit_next import Workspace, TextFormatReader, TextFormatParser
from core.interfaces.cb_model import CBModel
from registry.registry_fs import FSModelStore

def build_adf_string(
    context: Dict[str, Any],
    actions: Sequence[Dict[str, Any]],
    label_info: Optional[Tuple[int, float, float]] = None
) -> str:
    ctx_feats = " ".join(f"{k}={v}" for k, v in context.items())
    lines = [f"shared |C {ctx_feats}"]
    for idx, a in enumerate(actions):
        prefix = ""
        if label_info:
            cid, cost, prob = label_info
            prefix = f"{cid}:{cost}:{prob} "
        feats = " ".join(f"{k}={v}" for k, v in a.items())
        lines.append(f"{prefix}|A {feats}")
    return "\n".join(lines) + "\n"

class VWNextCBModel(CBModel):
    def __init__(self, **model_params: Any) -> None:
        vw_args = model_params.get('vw_args')
        if not vw_args or ('--cb_adf' not in vw_args and '--cb_explore_adf' not in vw_args):
            raise ValueError("Must pass vw_args including --cb_adf or --cb_explore_adf")
        self._vw_args = vw_args
        self.vw = Workspace(vw_args)
        self._parser = TextFormatParser(self.vw)
        self._last_actions = None

    def get_exploration_strategy(self) -> Any:
        return self._vw_args

    def batch_update(
        self,
        interactions: Optional[Sequence[Tuple[Any, Any, float, Optional[float]]]] = None,
        data_file: Optional[str] = None
    ) -> None:
        """
        Batch-style update from either:
          - `data_file` (VW-format ADF file), or
          - in-memory `interactions` list of (context, action_id, reward, prob)
        """
        if data_file is not None:
            with open(data_file, 'r') as f, TextFormatReader(self.vw, f) as reader:
                for example in reader:
                    self.vw.learn_one(example)
        elif interactions is not None:
            for interaction in interactions:
                self.update(interaction)
        else:
            raise ValueError("batch_update requires either interactions or data_file")

    def predict(
        self,
        row: Tuple[Dict[str, Any], Sequence[Dict[str, Any]]],
        eval_mode: bool = False
    ) -> Tuple[int, float]:
        context, actions = row
        self._last_actions = actions
        text = build_adf_string(context, actions)
        examples = [self._parser.parse_line(l) for l in text.strip().split('\n')]
        result = self.vw.predict_one(examples, eval_mode=eval_mode)
        preds = result[0] if isinstance(result, tuple) else result
        idx, score = preds[0]
        return idx, float(score)

    def batch_predict(
        self,
        rows: Sequence[Tuple[Dict[str, Any], Sequence[Dict[str, Any]]]],
        eval_mode: bool = False
    ) -> Sequence[Tuple[int, float]]:
        return [self.predict(row, eval_mode=eval_mode) for row in rows]

    def update(self, interaction: Tuple[Any, int, float, Optional[float]]) -> None:
        """
        Online update: uses last-predicted candidate_actions.
        """
        context, action_id, reward, prob_logged = interaction
        actions = self._last_actions
        if actions is None:
            raise ValueError("No last actions available for update()")
        cost = -reward
        prob = prob_logged or 1.0
        text = build_adf_string(context, actions, (action_id, cost, prob))
        examples = [self._parser.parse_line(l) for l in text.strip().split('\n')]
        for ex in examples:
            self.vw.learn_one(ex)

    def reset(self) -> None:
        self.vw = Workspace(self._vw_args)
        self._parser = TextFormatParser(self.vw)
        self._last_actions = None

    def save(
        self,
        name: str,
        version: str,
        registry_root: str,
        artifact_subpath: str = "models"
    ) -> str:
        store = FSModelStore()
        
        return store.save(name, self, version, registry_root, artifact_subpath)

    @classmethod
    def load(
        cls,
        name: str,
        version: str,
        registry_root: str,
        artifact_subpath: str = "models"
    ) -> 'VWNextCBModel':
        store = FSModelStore()
        model = store.load(name, version, registry_root, artifact_subpath)
        if not isinstance(model, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(model)}")
        return model
