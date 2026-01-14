import json
import csv
import os
from datetime import datetime
from typing import Dict, Any, List
from src.optimiser.individual import _json_default


class ExperimentLogger:
    def __init__(self, experiment_name: str, output_dir: str = 'results'):
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.experiment_dir = os.path.join(output_dir, experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)

        self.start_time = datetime.now()
        self.metadata = {
            'experiment_name': experiment_name,
            'start_time': self.start_time.isoformat()
        }

        self.jsonl_file = os.path.join(self.experiment_dir, 'evolution.jsonl')
        self.csv_file = os.path.join(self.experiment_dir, 'summary.csv')

    def log_generation(self, generation: int, stats: Dict[str, Any]):
        entry = {
            'generation': generation,
            'timestamp': datetime.now().isoformat(),
            **stats
        }

        with open(self.jsonl_file, 'a') as f:
            f.write(json.dumps(entry, default=_json_default) + '\n')

    def save_final_results(self, result: Dict[str, Any]):
        result_file = os.path.join(self.experiment_dir, 'final_result.json')
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2, default=_json_default)

    def save_best_config(self, config: Dict[str, Any], metrics: Dict[str, Any]):
        config_file = os.path.join(self.experiment_dir, 'best_config.json')
        data = {
            'config': config,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        with open(config_file, 'w') as f:
            json.dump(data, f, indent=2, default=_json_default)

    def export_to_csv(self, history: Dict[str, List]):
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Generation', 'Best Fitness', 'Mean Fitness'])

            for i in range(len(history['generations'])):
                writer.writerow([
                    history['generations'][i],
                    history['best_fitness'][i],
                    history['mean_fitness'][i]
                ])

    def get_experiment_dir(self) -> str:
        return self.experiment_dir
