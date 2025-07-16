# registry_fs.py

import os
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

        # Persist via joblib (handles numpy arrays, bytes, VW workspaces, etc.)
        joblib.dump(model, filepath)

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

        # Load and return the model
        return joblib.load(filepath)
