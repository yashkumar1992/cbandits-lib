# core/actions/adf_actions.py

from typing import Sequence, Dict, Any, Optional, Tuple, List
from core.interfaces.action import Action

class ADFActions:
    """
    Encapsulates context + actions and encodes VW-ADF text lines.
    """

    def __init__(
        self,
        context_features: Dict[str, Any],
        actions: Sequence[Action]
    ) -> None:
        """
        :param context_features: mapping from feature name to value
                                 (use '=' for strings, ':' for numerics)
        :param actions:          list of Action instances
        """
        self.context_features = context_features
        self.actions = list(actions)

    def record_stats(self, reward: float, **info: Any) -> None:
        """
        Record statistics for all actions (delegates to action.record_stats).
        """
        for action in self.actions:
            action.record_stats(reward=reward, **info)

    @staticmethod
    def _format_kv(k: str, v: Any) -> str:
        # '=' for str, ':' for others
        return f"{k}={v}" if isinstance(v, str) else f"{k}:{v}"

    def encode_lines(
        self,
        label_info: Optional[Tuple[Any, float, Optional[float]]] = None
    ) -> List[str]:
        """
        Generate VW ADF multiline text:

        - First line: "shared |C key=val key2:val2 ..."
        - Next lines: per-action, optionally labeled.

        :param label_info: (chosen_id, reward, prob)
        """
        lines: List[str] = []
        # shared line
        shared_parts = [
            self._format_kv(k, v) 
            for k, v in self.context_features.items()
        ]
        lines.append("shared |C " + " ".join(shared_parts))

        chosen_id = label_info[0] if label_info else None
        reward = label_info[1] if label_info else None
        prob = label_info[2] if label_info and label_info[2] is not None else 1.0

        # per-action lines
        for idx, action in enumerate(self.actions):
            feats = action.features()  # namespace->dict
            ns_parts = []
            for ns, fd in feats.items():
                kvs = [ self._format_kv(k, v) for k, v in fd.items() ]
                ns_parts.append(f"|{ns} " + " ".join(kvs))
            feat_line = " ".join(ns_parts)
            if action.get_id() == chosen_id:
                cost = -reward
                lines.append(f"{idx}:{cost}:{prob} " + feat_line)
            else:
                lines.append(feat_line)
        return lines
