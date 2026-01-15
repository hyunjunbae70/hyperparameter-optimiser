from typing import Dict, Any
from src.evaluation.metrics import MetricsCalculator


class MultiObjectiveFitness:
    def __init__(
        self,
        w_accuracy: float = 1.0,
        w_stability: float = 0.1,
        w_runtime: float = 0.05,
        reference_time: float = 100.0,
    ):
        self.w_accuracy = w_accuracy
        self.w_stability = w_stability
        self.w_runtime = w_runtime
        self.reference_time = reference_time

    def calculate_fitness(self, training_history: Dict[str, Any]) -> float:
        metrics = MetricsCalculator.get_all_metrics(
            training_history, self.reference_time
        )

        accuracy = metrics["accuracy"]
        stability_penalty = metrics["stability_penalty"]
        runtime_penalty = metrics["runtime_penalty"]

        fitness = (
            self.w_accuracy * accuracy
            - self.w_stability * stability_penalty
            - self.w_runtime * runtime_penalty
        )

        return fitness

    def calculate_with_metrics(
        self, training_history: Dict[str, Any]
    ) -> Dict[str, Any]:
        fitness = self.calculate_fitness(training_history)
        metrics = MetricsCalculator.get_all_metrics(
            training_history, self.reference_time
        )

        return {
            "fitness": fitness,
            "accuracy": metrics["accuracy"],
            "stability_penalty": metrics["stability_penalty"],
            "runtime_penalty": metrics["runtime_penalty"],
            "training_time": metrics["training_time"],
            "epochs_trained": metrics["epochs_trained"],
            "best_val_acc": metrics["best_val_acc"],
            "best_val_loss": metrics["best_val_loss"],
        }
