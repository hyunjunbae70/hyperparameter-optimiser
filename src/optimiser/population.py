from typing import List, Optional
import numpy as np
from src.optimiser.individual import Individual
from src.search_space.hyperparameters import HyperparameterSpace


class Population:
    def __init__(self, size: int, search_space: HyperparameterSpace):
        self.size = size
        self.search_space = search_space
        self.individuals: List[Individual] = []
        self.generation = 0

    def initialize_random(self):
        self.individuals = []
        for _ in range(self.size):
            config = self.search_space.sample_random_config()
            individual = Individual(config, generation=0)
            self.individuals.append(individual)
        self.generation = 0

    def initialize_with_default(self):
        self.individuals = []
        default_config = self.search_space.get_default_config()
        self.individuals.append(Individual(default_config, generation=0))

        for _ in range(self.size - 1):
            config = self.search_space.sample_random_config()
            individual = Individual(config, generation=0)
            self.individuals.append(individual)
        self.generation = 0

    def add_individual(self, individual: Individual):
        self.individuals.append(individual)

    def remove_individual(self, individual: Individual):
        self.individuals.remove(individual)

    def get_best(self, n: int = 1) -> List[Individual]:
        evaluated = [ind for ind in self.individuals if ind.is_evaluated()]
        if not evaluated:
            return []
        sorted_pop = sorted(evaluated, reverse=True)
        return sorted_pop[:n]

    def get_worst(self, n: int = 1) -> List[Individual]:
        evaluated = [ind for ind in self.individuals if ind.is_evaluated()]
        if not evaluated:
            return []
        sorted_pop = sorted(evaluated, reverse=False)
        return sorted_pop[:n]

    def get_unevaluated(self) -> List[Individual]:
        return [ind for ind in self.individuals if not ind.is_evaluated()]

    def get_evaluated(self) -> List[Individual]:
        return [ind for ind in self.individuals if ind.is_evaluated()]

    def replace_worst(self, new_individuals: List[Individual], elite_size: int = 0):
        evaluated = self.get_evaluated()
        if len(evaluated) < elite_size:
            elite_size = len(evaluated)

        elite = self.get_best(elite_size)

        num_to_replace = min(len(new_individuals), self.size - elite_size)
        self.individuals = elite + new_individuals[:num_to_replace]

        if len(self.individuals) < self.size:
            remaining = self.size - len(self.individuals)
            self.individuals.extend(
                new_individuals[num_to_replace : num_to_replace + remaining]
            )

    def get_statistics(self) -> dict:
        evaluated = self.get_evaluated()
        if not evaluated:
            return {
                "size": len(self.individuals),
                "evaluated": 0,
                "best_fitness": None,
                "mean_fitness": None,
                "std_fitness": None,
                "worst_fitness": None,
            }

        fitnesses = [ind.fitness for ind in evaluated]
        return {
            "size": len(self.individuals),
            "evaluated": len(evaluated),
            "best_fitness": max(fitnesses),
            "mean_fitness": np.mean(fitnesses),
            "std_fitness": np.std(fitnesses),
            "worst_fitness": min(fitnesses),
            "generation": self.generation,
        }

    def increment_generation(self):
        self.generation += 1

    def __len__(self) -> int:
        return len(self.individuals)

    def __iter__(self):
        return iter(self.individuals)

    def __getitem__(self, index):
        return self.individuals[index]
