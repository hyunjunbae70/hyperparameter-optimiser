import yaml
import os
from typing import Dict, Any


class Config:
    def __init__(self, config_file: str = None):
        self.config = self._load_default_config()

        if config_file and os.path.exists(config_file):
            with open(config_file, "r") as f:
                user_config = yaml.safe_load(f)
                self._merge_configs(self.config, user_config)

    def _load_default_config(self) -> Dict[str, Any]:
        return {
            "evolution": {
                "population_size": 30,
                "num_generations": 20,
                "crossover_rate": 0.8,
                "mutation_rate": 0.2,
                "elite_size": 3,
            },
            "training": {
                "max_epochs": 30,
                "early_stopping_patience": 5,
                "device": "cpu",
            },
            "parallel": {"num_workers": None},
            "data": {"data_dir": "./data", "val_split": 0.1},
            "fitness": {
                "w_accuracy": 1.0,
                "w_stability": 0.1,
                "w_runtime": 0.05,
                "reference_time": 100.0,
            },
            "experiment": {
                "name": "cifar10_hpo",
                "output_dir": "results",
                "use_cache": True,
            },
        }

    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]):
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self.config.get(section, {}).get(key, default)

    def set(self, section: str, key: str, value: Any):
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def to_dict(self) -> Dict[str, Any]:
        return self.config.copy()
