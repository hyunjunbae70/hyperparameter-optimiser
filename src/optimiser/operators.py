import numpy as np
from typing import Dict, Any, List, Tuple
from copy import deepcopy
from src.search_space.hyperparameters import HyperparameterSpace, ParamType
from src.optimiser.individual import Individual


class GeneticOperators:
    def __init__(self, search_space: HyperparameterSpace, mutation_rate: float = 0.2):
        self.search_space = search_space
        self.base_mutation_rate = mutation_rate
        self.mutation_rate = mutation_rate

    def mutate(self, individual: Individual, generation: int) -> Individual:
        mutated_config = deepcopy(individual.config)

        for param_name in self.search_space.get_all_params():
            if np.random.random() < self.mutation_rate:
                mutated_config[param_name] = self._mutate_param(
                    param_name,
                    mutated_config[param_name],
                    generation
                )

        mutated_config = self.search_space.validate_config(mutated_config)
        return Individual(mutated_config, generation)

    def _mutate_param(self, param_name: str, value: Any, generation: int) -> Any:
        param_info = self.search_space.get_param_info(param_name)
        param_type = param_info['type']

        if param_type == ParamType.CONTINUOUS:
            return self._gaussian_mutation(param_name, value, generation)
        elif param_type in [ParamType.DISCRETE, ParamType.CATEGORICAL]:
            return self._uniform_mutation(param_name, value)
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")

    def _gaussian_mutation(self, param_name: str, value: float, generation: int) -> float:
        param_info = self.search_space.get_param_info(param_name)
        min_val, max_val = param_info['range']
        is_log_scale = param_info.get('log_scale', False)

        decay_factor = 1.0 / (1.0 + 0.01 * generation)

        if is_log_scale:
            log_value = np.log10(value)
            log_min, log_max = np.log10(min_val), np.log10(max_val)
            sigma = (log_max - log_min) * 0.1 * decay_factor
            new_log_value = log_value + np.random.normal(0, sigma)
            new_value = 10 ** new_log_value
        else:
            sigma = (max_val - min_val) * 0.1 * decay_factor
            new_value = value + np.random.normal(0, sigma)

        return float(np.clip(new_value, min_val, max_val))

    def _uniform_mutation(self, param_name: str, value: Any) -> Any:
        param_info = self.search_space.get_param_info(param_name)
        choices = param_info['choices']

        other_choices = [c for c in choices if c != value]
        if other_choices:
            return np.random.choice(other_choices)
        return value

    def crossover(self, parent1: Individual, parent2: Individual, generation: int) -> Tuple[Individual, Individual]:
        child1_config = {}
        child2_config = {}

        for param_name in self.search_space.get_all_params():
            param_info = self.search_space.get_param_info(param_name)
            param_type = param_info['type']

            p1_value = parent1.config[param_name]
            p2_value = parent2.config[param_name]

            if param_type == ParamType.CONTINUOUS:
                c1_value, c2_value = self._sbx_crossover(param_name, p1_value, p2_value)
            else:
                c1_value, c2_value = self._uniform_crossover(p1_value, p2_value)

            child1_config[param_name] = c1_value
            child2_config[param_name] = c2_value

        child1_config = self.search_space.validate_config(child1_config)
        child2_config = self.search_space.validate_config(child2_config)

        return (Individual(child1_config, generation),
                Individual(child2_config, generation))

    def _sbx_crossover(self, param_name: str, p1_value: float, p2_value: float,
                       eta: float = 15.0) -> Tuple[float, float]:
        param_info = self.search_space.get_param_info(param_name)
        min_val, max_val = param_info['range']

        if abs(p1_value - p2_value) < 1e-9:
            return p1_value, p2_value

        if p1_value > p2_value:
            p1_value, p2_value = p2_value, p1_value

        u = np.random.random()

        if u <= 0.5:
            beta = (2.0 * u) ** (1.0 / (eta + 1.0))
        else:
            beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))

        c1_value = 0.5 * ((p1_value + p2_value) - beta * (p2_value - p1_value))
        c2_value = 0.5 * ((p1_value + p2_value) + beta * (p2_value - p1_value))

        c1_value = np.clip(c1_value, min_val, max_val)
        c2_value = np.clip(c2_value, min_val, max_val)

        return float(c1_value), float(c2_value)

    def _uniform_crossover(self, p1_value: Any, p2_value: Any) -> Tuple[Any, Any]:
        if np.random.random() < 0.5:
            return p1_value, p2_value
        else:
            return p2_value, p1_value

    def tournament_selection(self, population: List[Individual], k: int = 3) -> Individual:
        tournament = np.random.choice(population, size=min(k, len(population)), replace=False)
        return max(tournament, key=lambda ind: ind.fitness if ind.fitness is not None else float('-inf'))

    def select_parents(self, population: List[Individual], num_pairs: int) -> List[Tuple[Individual, Individual]]:
        pairs = []
        for _ in range(num_pairs):
            parent1 = self.tournament_selection(population)
            parent2 = self.tournament_selection(population)
            pairs.append((parent1, parent2))
        return pairs

    def adapt_mutation_rate(self, generation: int, max_generations: int):
        progress = generation / max_generations
        self.mutation_rate = self.base_mutation_rate * (1.0 - 0.5 * progress)
