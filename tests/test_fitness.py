import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.fitness import MultiObjectiveFitness


def test_fitness_calculation():
    fitness_calc = MultiObjectiveFitness(
        w_accuracy=1.0, w_stability=0.1, w_runtime=0.05
    )

    training_history = {
        "best_val_acc": 0.85,
        "val_loss_std": 0.05,
        "training_time": 100.0,
        "epochs_trained": 25,
    }

    result = fitness_calc.calculate_with_metrics(training_history)

    assert "fitness" in result
    assert "accuracy" in result
    assert "stability_penalty" in result
    assert "runtime_penalty" in result
    assert isinstance(result["fitness"], float)


def test_fitness_higher_accuracy_better():
    fitness_calc = MultiObjectiveFitness()

    history1 = {
        "best_val_acc": 0.9,
        "val_loss_std": 0.05,
        "training_time": 100.0,
        "epochs_trained": 25,
    }

    history2 = {
        "best_val_acc": 0.7,
        "val_loss_std": 0.05,
        "training_time": 100.0,
        "epochs_trained": 25,
    }

    fitness1 = fitness_calc.calculate_fitness(history1)
    fitness2 = fitness_calc.calculate_fitness(history2)

    assert fitness1 > fitness2
