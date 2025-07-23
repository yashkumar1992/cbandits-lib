import os
import joblib

class FSModelStore:
    """
    Filesystem-backed model store for LinUCBModel.
    Stores the model as joblib dump of its internal numpy structures and metadata.
    """

    def save(
        self,
        name: str,
        model_data: dict,
        version: str,
        registry_root: str,
        artifact_subpath: str = "models",
    ) -> str:
        """
        Serialize LinUCBModel weights and persist to disk via joblib.
        """
        dirpath = os.path.join(registry_root, name, version, artifact_subpath)
        os.makedirs(dirpath, exist_ok=True)

        filename = f"{name}-{version}.pkl"
        filepath = os.path.join(dirpath, filename)

        

        joblib.dump(model_data, filepath)
        return filepath

    def load(
        self,
        name: str,
        version: str,
        registry_root: str,
        artifact_subpath: str = "models",
    ) -> dict:
        """
        Load model weights from disk as a dictionary.
        You can then pass this to LinUCBModel constructor.
        """
        model_dir = os.path.join(registry_root, name)
        if version is None:
            versions = sorted(os.listdir(model_dir))
            version = versions[-1]

        filename = f"{name}-{version}.pkl"
        filepath = os.path.join(model_dir, version, artifact_subpath, filename)

        model_data = joblib.load(filepath)
        return model_data
