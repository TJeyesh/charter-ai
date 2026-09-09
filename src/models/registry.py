"""
Charter-AI — Model Registry.

Tracks trained model versions, their metrics, and artifacts.
Enables reproducibility and safe model rollback.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """
    Local model registry backed by the trained_models/ directory
    and the model_registry database table.

    Directory structure:
        trained_models/
        ├── freight_forecaster/
        │   ├── v0.1.0/
        │   │   ├── model.joblib
        │   │   └── metadata.json
        │   └── v0.2.0/
        │       ├── model.joblib
        │       └── metadata.json
        └── idle_time_predictor/
            └── v0.1.0/
                ├── model.joblib
                └── metadata.json
    """

    def __init__(self, base_dir: Optional[str] = None):
        settings = get_settings()
        self.base_dir = Path(base_dir or settings.trained_models_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_model(
        self,
        model: Any,
        model_name: str,
        model_version: str,
        metrics: Dict[str, float],
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save a trained model with its metadata.

        Args:
            model: Trained model object (must be joblib-serializable).
            model_name: E.g. "freight_forecaster", "idle_time_predictor".
            model_version: Semantic version string, e.g. "v0.1.0".
            metrics: Evaluation metrics dict.
            hyperparameters: Training hyperparameters.

        Returns:
            Path to the saved model artifact.
        """
        # TODO: Implement joblib serialization + metadata.json
        raise NotImplementedError("Model saving will be implemented in the ML phase.")

    def load_model(
        self,
        model_name: str,
        model_version: Optional[str] = None,
    ) -> Any:
        """
        Load a model by name and version.

        If version is None, loads the latest active version.

        Args:
            model_name: Model identifier.
            model_version: Specific version, or None for latest.

        Returns:
            Deserialized model object.
        """
        # TODO: Implement model loading with version resolution
        raise NotImplementedError("Model loading will be implemented in the ML phase.")

    def list_versions(self, model_name: str) -> list:
        """List all available versions of a model."""
        model_dir = self.base_dir / model_name
        if not model_dir.exists():
            return []
        return sorted([d.name for d in model_dir.iterdir() if d.is_dir()])

    def get_active_version(self, model_name: str) -> Optional[str]:
        """Get the currently active (production) version of a model."""
        # TODO: Query model_registry table for is_active=True
        raise NotImplementedError
