import json
import os
import torch
from typing import Dict, Any
from src.optimiser.individual import _json_default


class ResultsManager:
    def __init__(self, experiment_dir: str):
        self.experiment_dir = experiment_dir
        os.makedirs(experiment_dir, exist_ok=True)

    def save_model_checkpoint(self, model: torch.nn.Module, filename: str = 'best_model.pth'):
        checkpoint_path = os.path.join(self.experiment_dir, filename)
        torch.save(model.state_dict(), checkpoint_path)
        return checkpoint_path

    def save_training_history(self, history: Dict[str, Any], filename: str = 'training_history.json'):
        history_path = os.path.join(self.experiment_dir, filename)
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2, default=_json_default)
        return history_path

    def save_hyperparameters(self, config: Dict[str, Any], filename: str = 'hyperparameters.json'):
        config_path = os.path.join(self.experiment_dir, filename)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=_json_default)
        return config_path

    def load_results(self) -> Dict[str, Any]:
        results = {}

        final_result_path = os.path.join(self.experiment_dir, 'final_result.json')
        if os.path.exists(final_result_path):
            with open(final_result_path, 'r') as f:
                results['final_result'] = json.load(f)

        best_config_path = os.path.join(self.experiment_dir, 'best_config.json')
        if os.path.exists(best_config_path):
            with open(best_config_path, 'r') as f:
                results['best_config'] = json.load(f)

        return results

    def generate_report(self, result: Dict[str, Any]) -> str:
        report_lines = []
        report_lines.append("="*60)
        report_lines.append("Hyperparameter Optimisation Results")
        report_lines.append("="*60)
        report_lines.append("")

        if 'best_fitness' in result:
            report_lines.append(f"Best Fitness: {result['best_fitness']:.4f}")

        if 'best_metrics' in result:
            report_lines.append("\nBest Metrics:")
            for key, value in result['best_metrics'].items():
                if isinstance(value, (int, float)):
                    report_lines.append(f"  {key}: {value:.4f}")
                else:
                    report_lines.append(f"  {key}: {value}")

        if 'best_config' in result:
            report_lines.append("\nBest Configuration:")
            for key, value in result['best_config'].items():
                report_lines.append(f"  {key}: {value}")

        if 'total_time' in result:
            report_lines.append(f"\nTotal Optimisation Time: {result['total_time']:.2f}s")

        if 'cache_stats' in result:
            report_lines.append("\nCache Statistics:")
            for key, value in result['cache_stats'].items():
                if isinstance(value, (int, float)):
                    report_lines.append(f"  {key}: {value:.4f}")
                else:
                    report_lines.append(f"  {key}: {value}")

        report_lines.append("="*60)

        report_text = "\n".join(report_lines)

        report_path = os.path.join(self.experiment_dir, 'report.txt')
        with open(report_path, 'w') as f:
            f.write(report_text)

        return report_text
