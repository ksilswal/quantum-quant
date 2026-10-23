"""Importing this package registers every built-in algorithm."""
from . import montecarlo, predictive, statistical  # noqa: F401  (registration side effects)
from .base import REGISTRY, AlgoOutput, Algorithm, Param, RunContext, catalog, get, register, resolve_params

__all__ = [
    "REGISTRY", "AlgoOutput", "Algorithm", "Param", "RunContext",
    "catalog", "get", "register", "resolve_params",
]
