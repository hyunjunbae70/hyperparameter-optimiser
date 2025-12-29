import numpy as np
from typing import Dict, Any, List, Tuple
from enum import Enum


class ParamType(Enum):
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"


class HyperparameterSpace:
    def __init__(self):
        self.params = {
            'learning_rate': {
                'type': ParamType.CONTINUOUS,
                'range': (1e-5, 1e-1),
                'log_scale': True,
                'default': 0.001
            },
            'optimiser': {
                'type': ParamType.CATEGORICAL,
                'choices': ['SGD', 'Adam', 'AdamW', 'RMSprop'],
                'default': 'Adam'
            },
            'batch_size': {
                'type': ParamType.DISCRETE,
                'choices': [32, 64, 128, 256],
                'default': 64
            },
            'dropout_rate': {
                'type': ParamType.CONTINUOUS,
                'range': (0.0, 0.5),
                'log_scale': False,
                'default': 0.2
            },
            'num_layers': {
                'type': ParamType.DISCRETE,
                'choices': [2, 3, 4, 5],
                'default': 3
            },
            'base_channels': {
                'type': ParamType.DISCRETE,
                'choices': [32, 64, 128],
                'default': 64
            },
            'weight_decay': {
                'type': ParamType.CONTINUOUS,
                'range': (1e-6, 1e-2),
                'log_scale': True,
                'default': 1e-4
            },
            'momentum': {
                'type': ParamType.CONTINUOUS,
                'range': (0.8, 0.99),
                'log_scale': False,
                'default': 0.9
            }
        }

    def get_param_info(self, param_name: str) -> Dict[str, Any]:
        if param_name not in self.params:
            raise ValueError(f"Unknown parameter: {param_name}")
        return self.params[param_name]

    def get_all_params(self) -> List[str]:
        return list(self.params.keys())

    def sample_random(self, param_name: str) -> Any:
        param_info = self.get_param_info(param_name)
        param_type = param_info['type']

        if param_type == ParamType.CONTINUOUS:
            min_val, max_val = param_info['range']
            if param_info.get('log_scale', False):
                log_min, log_max = np.log10(min_val), np.log10(max_val)
                value = 10 ** np.random.uniform(log_min, log_max)
            else:
                value = np.random.uniform(min_val, max_val)
            return float(value)

        elif param_type in [ParamType.DISCRETE, ParamType.CATEGORICAL]:
            return np.random.choice(param_info['choices'])

        else:
            raise ValueError(f"Unknown parameter type: {param_type}")

    def sample_random_config(self) -> Dict[str, Any]:
        config = {}
        for param_name in self.get_all_params():
            config[param_name] = self.sample_random(param_name)
        return config

    def get_default_config(self) -> Dict[str, Any]:
        config = {}
        for param_name, param_info in self.params.items():
            config[param_name] = param_info['default']
        return config

    def clip_value(self, param_name: str, value: Any) -> Any:
        param_info = self.get_param_info(param_name)
        param_type = param_info['type']

        if param_type == ParamType.CONTINUOUS:
            min_val, max_val = param_info['range']
            return float(np.clip(value, min_val, max_val))

        elif param_type == ParamType.DISCRETE:
            choices = param_info['choices']
            if isinstance(value, (int, float)):
                idx = np.argmin([abs(value - c) for c in choices])
                return choices[idx]
            return value if value in choices else choices[0]

        elif param_type == ParamType.CATEGORICAL:
            choices = param_info['choices']
            return value if value in choices else choices[0]

        else:
            raise ValueError(f"Unknown parameter type: {param_type}")

    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        validated = {}
        for param_name in self.get_all_params():
            if param_name in config:
                validated[param_name] = self.clip_value(param_name, config[param_name])
            else:
                validated[param_name] = self.params[param_name]['default']
        return validated
