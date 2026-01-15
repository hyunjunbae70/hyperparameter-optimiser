import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from typing import Dict, Any, List


class Visualiser:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        sns.set_style("whitegrid")

    def plot_fitness_evolution(self, history: Dict[str, List], save_path: str = None):
        if save_path is None:
            save_path = os.path.join(self.output_dir, "fitness_evolution.png")

        fig, ax = plt.subplots(figsize=(10, 6))

        generations = history["generations"]
        best_fitness = history["best_fitness"]
        mean_fitness = history["mean_fitness"]

        ax.plot(
            generations,
            best_fitness,
            "b-o",
            label="Best Fitness",
            linewidth=2,
            markersize=6,
        )
        ax.plot(
            generations,
            mean_fitness,
            "r--s",
            label="Mean Fitness",
            linewidth=2,
            markersize=4,
        )

        ax.set_xlabel("Generation", fontsize=12)
        ax.set_ylabel("Fitness", fontsize=12)
        ax.set_title(
            "Fitness Evolution Over Generations", fontsize=14, fontweight="bold"
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        return save_path

    def plot_parameter_distribution(
        self, best_configs: List[Dict], param_name: str, save_path: str = None
    ):
        if save_path is None:
            save_path = os.path.join(self.output_dir, f"param_dist_{param_name}.png")

        values = [
            config["config"][param_name]
            for config in best_configs
            if param_name in config["config"]
        ]

        if not values:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        if isinstance(values[0], (int, float)):
            ax.plot(range(len(values)), values, "g-o", linewidth=2, markersize=6)
            ax.set_ylabel(f"{param_name}", fontsize=12)
        else:
            unique_values = list(set(values))
            value_counts = [values.count(v) for v in unique_values]
            ax.bar(range(len(unique_values)), value_counts)
            ax.set_xticks(range(len(unique_values)))
            ax.set_xticklabels(unique_values, rotation=45, ha="right")
            ax.set_ylabel("Frequency", fontsize=12)

        ax.set_xlabel("Generation", fontsize=12)
        ax.set_title(f"{param_name} Evolution", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        return save_path

    def plot_training_curves(self, history: Dict[str, List], save_path: str = None):
        if save_path is None:
            save_path = os.path.join(self.output_dir, "training_curves.png")

        train_loss = history.get("train_loss", [])
        val_loss = history.get("val_loss", [])
        train_acc = history.get("train_acc", [])
        val_acc = history.get("val_acc", [])

        if not train_loss:
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        epochs = range(1, len(train_loss) + 1)

        ax1.plot(epochs, train_loss, "b-", label="Train Loss", linewidth=2)
        ax1.plot(epochs, val_loss, "r-", label="Val Loss", linewidth=2)
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Loss", fontsize=12)
        ax1.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, train_acc, "b-", label="Train Accuracy", linewidth=2)
        ax2.plot(epochs, val_acc, "r-", label="Val Accuracy", linewidth=2)
        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel("Accuracy", fontsize=12)
        ax2.set_title(
            "Training and Validation Accuracy", fontsize=14, fontweight="bold"
        )
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        return save_path

    def plot_multi_objective_tradeoff(
        self, best_configs: List[Dict], save_path: str = None
    ):
        if save_path is None:
            save_path = os.path.join(self.output_dir, "multi_objective_tradeoff.png")

        accuracies = []
        stabilities = []
        runtimes = []

        for config in best_configs:
            if "metrics" in config and config["metrics"]:
                metrics = config["metrics"]
                accuracies.append(metrics.get("accuracy", 0))
                stabilities.append(metrics.get("stability_penalty", 0))
                runtimes.append(metrics.get("training_time", 0))

        if not accuracies:
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        scatter1 = ax1.scatter(
            accuracies,
            stabilities,
            c=range(len(accuracies)),
            cmap="viridis",
            s=100,
            alpha=0.6,
        )
        ax1.set_xlabel("Accuracy", fontsize=12)
        ax1.set_ylabel("Stability Penalty", fontsize=12)
        ax1.set_title("Accuracy vs Stability", fontsize=14, fontweight="bold")
        plt.colorbar(scatter1, ax=ax1, label="Generation")
        ax1.grid(True, alpha=0.3)

        scatter2 = ax2.scatter(
            accuracies,
            runtimes,
            c=range(len(accuracies)),
            cmap="viridis",
            s=100,
            alpha=0.6,
        )
        ax2.set_xlabel("Accuracy", fontsize=12)
        ax2.set_ylabel("Training Time (s)", fontsize=12)
        ax2.set_title("Accuracy vs Runtime", fontsize=14, fontweight="bold")
        plt.colorbar(scatter2, ax=ax2, label="Generation")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        return save_path

    def generate_all_plots(self, result: Dict[str, Any]):
        plots = {}

        if "history" in result:
            history = result["history"]

            plots["fitness_evolution"] = self.plot_fitness_evolution(history)

            if "best_configs" in history:
                best_configs = history["best_configs"]

                param_names = [
                    "learning_rate",
                    "batch_size",
                    "dropout_rate",
                    "num_layers",
                    "optimiser",
                ]
                for param_name in param_names:
                    plot_path = self.plot_parameter_distribution(
                        best_configs, param_name
                    )
                    if plot_path:
                        plots[f"param_{param_name}"] = plot_path

                plots["multi_objective"] = self.plot_multi_objective_tradeoff(
                    best_configs
                )

        return plots
