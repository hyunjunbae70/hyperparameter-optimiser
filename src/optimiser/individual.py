import hashlib
import json
from typing import Dict, Any, Optional
from copy import deepcopy

import numpy as np


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class Individual:
    def __init__(self, config: Dict[str, Any], generation: int = 0):
        self.config = deepcopy(config)
        self.generation = generation
        self.fitness: Optional[float] = None
        self.metrics: Dict[str, Any] = {}
        self._config_hash: Optional[str] = None

    @property
    def config_hash(self) -> str:
        if self._config_hash is None:
            self._config_hash = self._compute_hash()
        return self._config_hash

    def _compute_hash(self) -> str:
        config_str = json.dumps(self.config, sort_keys=True, default=_json_default)
        return hashlib.sha256(config_str.encode()).hexdigest()

    def set_fitness(self, fitness: float, metrics: Optional[Dict[str, Any]] = None):
        self.fitness = fitness
        if metrics is not None:
            self.metrics = deepcopy(metrics)

    def is_evaluated(self) -> bool:
        return self.fitness is not None

    def copy(self) -> "Individual":
        new_individual = Individual(self.config, self.generation)
        if self.is_evaluated():
            new_individual.set_fitness(self.fitness, self.metrics)
        return new_individual

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config,
            "generation": self.generation,
            "fitness": self.fitness,
            "metrics": self.metrics,
            "config_hash": self.config_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Individual":
        individual = cls(data["config"], data.get("generation", 0))
        if data.get("fitness") is not None:
            individual.set_fitness(data["fitness"], data.get("metrics", {}))
        return individual

    def __repr__(self) -> str:
        fitness_str = f"{self.fitness:.4f}" if self.fitness is not None else "N/A"
        return f"Individual(gen={self.generation}, fitness={fitness_str}, config={self.config})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Individual):
            return False
        return self.config_hash == other.config_hash

    def __hash__(self) -> int:
        return int(self.config_hash[:16], 16)

    def __lt__(self, other: "Individual") -> bool:
        if self.fitness is None:
            return False
        if other.fitness is None:
            return True
        return self.fitness > other.fitness
