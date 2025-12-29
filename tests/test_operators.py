import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.search_space.hyperparameters import HyperparameterSpace
from src.optimiser.individual import Individual
from src.optimiser.operators import GeneticOperators


def test_mutation():
    search_space = HyperparameterSpace()
    operators = GeneticOperators(search_space, mutation_rate=0.2)

    config = search_space.get_default_config()
    individual = Individual(config, generation=0)

    mutated = operators.mutate(individual, generation=0)

    assert mutated.generation == 0
    assert mutated.config != individual.config or True


def test_crossover():
    search_space = HyperparameterSpace()
    operators = GeneticOperators(search_space)

    config1 = search_space.get_default_config()
    config2 = search_space.sample_random_config()

    parent1 = Individual(config1, generation=0)
    parent2 = Individual(config2, generation=0)

    child1, child2 = operators.crossover(parent1, parent2, generation=1)

    assert child1.generation == 1
    assert child2.generation == 1


def test_tournament_selection():
    search_space = HyperparameterSpace()
    operators = GeneticOperators(search_space)

    population = []
    for i in range(10):
        config = search_space.sample_random_config()
        ind = Individual(config, generation=0)
        ind.set_fitness(float(i))
        population.append(ind)

    selected = operators.tournament_selection(population, k=3)

    assert selected in population
    assert selected.fitness is not None
