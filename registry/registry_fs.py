# registry_fs.py

import os

from vowpal_wabbit_next import TextFormatParser, Workspace
import joblib
from typing import Any

class FSModelStore:
    """
    Filesystem-backed ModelStore using joblib for serialization.
    Artifacts are saved as `{name}-{version}.joblib`.
    """

    def save(
        self,
        name: str,
        model: Any,
        version: str,
        registry_root: str,
        artifact_subpath: str = "models"
    ) -> str:
        # Build target directory: <registry_root>/<name>/<version>/<artifact_subpath>/
        dirpath = os.path.join(registry_root, name, version, artifact_subpath)
        os.makedirs(dirpath, exist_ok=True)

        # Filename ends with .joblib
        filename = f"{name}-{version}.joblib"
        filepath = os.path.join(dirpath, filename)
        # Serialize VW internals to bytes
        model_bytes = model.vw.serialize()  # :contentReference[oaicite:0]{index=0}
        #model_bytes = self.vw.serialize()
        save_obj = {
            "model_bytes": model_bytes,
        }
        #path = os.path.join(dirpath, f"{name}-{version}.joblib")
        joblib.dump(save_obj, filepath)


        # Persist via joblib (handles numpy arrays, bytes, VW workspaces, etc.)
        #joblib.dump(model_bytes, filepath)

        return filepath

    def load(
        self,
        name: str,
        version: str,
        registry_root: str,
        artifact_subpath: str = "models"
    ) -> Any:
        # If version=None, pick the numerically or lexically highest version folder
        if version is None:
            versions = sorted(os.listdir(os.path.join(registry_root, name)))
            version = versions[-1]

        filename = f"{name}-{version}.joblib"
        filepath = os.path.join(registry_root, name, version, artifact_subpath, filename)

        # Read back the raw VW bytes
        #model_bytes = joblib.load(filepath)

        save_obj = joblib.load(filepath)
        #vw_args = save_obj["vw_args"]
        model_bytes = save_obj["model_bytes"]

        # Rebuild adapter and attach workspace
        model.vw = Workspace(model_data=model_bytes)
        model._vw_args = ""
        
        model._parser = TextFormatParser(model.vw)
        model._last_actions = None
        return model
    
    

