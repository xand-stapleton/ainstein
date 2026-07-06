from __future__ import annotations

from pydantic import BaseModel

from .base import BaseConfig, KerasSerialisableObject, ModelSpecific


class SphereProperties(BaseModel, KerasSerialisableObject):
    ball: bool = True
    density_power: float = 1.0
    lorentzian: bool = False


class SphereConfig(BaseConfig, ModelSpecific):
    model_specific: SphereProperties = SphereProperties()
