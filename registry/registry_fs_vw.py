
import os
import joblib

from vowpal_wabbit_next import Workspace

class FSModelStore:
    """
    Filesystem-backed store for VWNextCBModel: serialize raw VW workspace bytes to disk and reload.
    No adapter imports to avoid circular dependencies.
    """

    def save(
        self,
        name: str,
        model: object,
        version: str,
        registry_root: str,
        artifact_subpath: str = "models",
    ) -> str:
        """
        Serialize model.vw workspace bytes and persist via joblib.
        """
        dirpath = os.path.join(registry_root, name, version, artifact_subpath)
        os.makedirs(dirpath, exist_ok=True)

        filename = f"{name}-{version}.joblib"
        filepath = os.path.join(dirpath, filename)

        # Only raw VW bytes
        model_bytes = model.serialize()
        joblib.dump(model_bytes, filepath)
        return filepath

    def load(
        self,
        name: str,
        version: str,
        registry_root: str,
        artifact_subpath: str = "models",
    ) -> bytes:
        """
        Load raw VW workspace bytes from disk.
        """
        if version is None:
            versions = sorted(os.listdir(os.path.join(registry_root, name)))
            version = versions[-1]

        filename = f"{name}-{version}.joblib"
        filepath = os.path.join(registry_root, name, version, artifact_subpath, filename)

        model_bytes = joblib.load(filepath)
        #vw_args = save_obj["vw_args"]
        
        return model_bytes
