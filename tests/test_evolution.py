import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.search_space.hyperparameters import HyperparameterSpace
from src.optimiser.population import Population
from src.optimiser.individual import Individual


def test_population_initialization():
    search_space = HyperparameterSpace()
    population = Population(size=10, search_space=search_space)

    population.initialize_random()

    assert len(population.individuals) == 10
    assert population.generation == 0


def test_population_best_selection():
    search_space = HyperparameterSpace()
    population = Population(size=10, search_space=search_space)
    population.initialize_random()

    for i, ind in enumerate(population.individuals):
        ind.set_fitness(float(i))

    best = population.get_best(n=3)

    assert len(best) == 3
    assert best[0].fitness == 9.0
    assert best[1].fitness == 8.0
    assert best[2].fitness == 7.0


def test_individual_hash():
    search_space = HyperparameterSpace()
    config = search_space.get_default_config()

    ind1 = Individual(config, generation=0)
    ind2 = Individual(config, generation=0)

    assert ind1.config_hash == ind2.config_hash
