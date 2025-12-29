from typing import Dict, Any
import numpy as np


class MetricsCalculator:
    @staticmethod
    def calculate_accuracy_score(history: Dict[str, Any]) -> float:
        return history.get('best_val_acc', 0.0)

    @staticmethod
    def calculate_stability_penalty(history: Dict[str, Any]) -> float:
        val_loss_std = history.get('val_loss_std', 0.0)
        return val_loss_std

    @staticmethod
    def calculate_runtime_penalty(history: Dict[str, Any], reference_time: float = 100.0) -> float:
        training_time = history.get('training_time', reference_time)
        normalised_time = training_time / reference_time
        return normalised_time

    @staticmethod
    def get_all_metrics(history: Dict[str, Any], reference_time: float = 100.0) -> Dict[str, float]:
        return {
            'accuracy': MetricsCalculator.calculate_accuracy_score(history),
            'stability_penalty': MetricsCalculator.calculate_stability_penalty(history),
            'runtime_penalty': MetricsCalculator.calculate_runtime_penalty(history, reference_time),
            'training_time': history.get('training_time', 0.0),
            'epochs_trained': history.get('epochs_trained', 0),
            'best_val_acc': history.get('best_val_acc', 0.0),
            'best_val_loss': history.get('best_val_loss', float('inf')),
            'final_val_acc': history.get('final_val_acc', 0.0),
            'final_val_loss': history.get('final_val_loss', float('inf'))
        }
