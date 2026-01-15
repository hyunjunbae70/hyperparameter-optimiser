import json
import os
from typing import Dict, Any, Optional
from src.optimiser.individual import _json_default


class EvaluationCache:
    def __init__(self, cache_file: str = "results/eval_cache.json"):
        self.cache_file = cache_file
        self.cache: Dict[str, Dict[str, Any]] = {}
        self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as f:
                    self.cache = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.cache = {}
        else:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            self.cache = {}

    def _save_cache(self):
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f, indent=2, default=_json_default)

    def get(self, config_hash: str) -> Optional[Dict[str, Any]]:
        return self.cache.get(config_hash)

    def put(self, config_hash: str, result: Dict[str, Any]):
        self.cache[config_hash] = result
        self._save_cache()

    def has(self, config_hash: str) -> bool:
        return config_hash in self.cache

    def clear(self):
        self.cache = {}
        self._save_cache()

    def size(self) -> int:
        return len(self.cache)

    def get_statistics(self) -> Dict[str, Any]:
        if not self.cache:
            return {"total_evaluations": 0, "cache_hits": 0, "cache_size": 0}

        return {"total_evaluations": len(self.cache), "cache_size": len(self.cache)}
