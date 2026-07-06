from __future__ import annotations

from pathlib import Path

import tensorflow as tf


class BaseVisualiser:
    def __init__(
        self, model_parent: Path | str | None = None, network_custom_objects: dict = {}
    ) -> None:
        self.model_parent = Path(model_parent)
        self.model_blob = self.model_parent / "final_model.keras"
        self.loaded_model = tf.keras.models.load_model(
            self.model_blob, custom_objects=network_custom_objects
        )
        # We load from the file because it's not necessarily true that the
        # state of the model at the current epoch is the best (i.e. the one
        # with the lowest loss)

        self.config = self.loaded_model.config
