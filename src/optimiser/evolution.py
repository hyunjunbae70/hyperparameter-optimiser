import os
import json
import time
from typing import Dict, Any, List
from src.search_space.hyperparameters import HyperparameterSpace
from src.optimiser.individual import _json_default
from src.optimiser.population import Population
from src.optimiser.operators import GeneticOperators
from src.parallel.evaluator import ParallelEvaluator


class EvolutionaryOptimiser:
    def __init__(self, population_size: int = 30, num_generations: int = 20,
                 crossover_rate: float = 0.8, mutation_rate: float = 0.2,
                 elite_size: int = 3, num_workers: int = None,
                 data_dir: str = './data', max_epochs: int = 30,
                 device: str = 'cpu', checkpoint_dir: str = 'results/checkpoints'):

        self.population_size = population_size
        self.num_generations = num_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size

        self.search_space = HyperparameterSpace()
        self.population = Population(population_size, self.search_space)
        self.operators = GeneticOperators(self.search_space, mutation_rate)
        self.evaluator = ParallelEvaluator(
            num_workers=num_workers,
            data_dir=data_dir,
            max_epochs=max_epochs,
            device=device
        )

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.history = {
            'generations': [],
            'best_fitness': [],
            'mean_fitness': [],
            'best_configs': []
        }

    def optimise(self, verbose: bool = True) -> Dict[str, Any]:
        start_time = time.time()

        if verbose:
            print(f"Initializing population of size {self.population_size}...")
        self.population.initialize_with_default()

        for generation in range(self.num_generations):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Generation {generation + 1}/{self.num_generations}")
                print(f"{'='*60}")

            self.evaluator.evaluate_population(self.population.individuals, verbose=verbose)

            stats = self.population.get_statistics()
            self.history['generations'].append(generation)
            self.history['best_fitness'].append(stats['best_fitness'])
            self.history['mean_fitness'].append(stats['mean_fitness'])

            best_ind = self.population.get_best(1)[0]
            self.history['best_configs'].append(best_ind.to_dict())

            if verbose:
                print(f"\nGeneration {generation + 1} Statistics:")
                print(f"  Best Fitness: {stats['best_fitness']:.4f}")
                print(f"  Mean Fitness: {stats['mean_fitness']:.4f}")
                print(f"  Std Fitness:  {stats['std_fitness']:.4f}")
                print(f"  Best Config: {best_ind.config}")

            if generation < self.num_generations - 1:
                offspring = []

                num_offspring_pairs = (self.population_size - self.elite_size) // 2

                parent_pairs = self.operators.select_parents(
                    self.population.individuals,
                    num_offspring_pairs
                )

                for parent1, parent2 in parent_pairs:
                    if len(offspring) < self.population_size - self.elite_size:
                        if self.operators.adapt_mutation_rate:
                            self.operators.adapt_mutation_rate(generation, self.num_generations)

                        if self.crossover_rate > 0 and len(offspring) < self.population_size - self.elite_size - 1:
                            child1, child2 = self.operators.crossover(parent1, parent2, generation + 1)
                            child1 = self.operators.mutate(child1, generation + 1)
                            child2 = self.operators.mutate(child2, generation + 1)
                            offspring.extend([child1, child2])
                        else:
                            child = self.operators.mutate(parent1.copy(), generation + 1)
                            offspring.append(child)

                while len(offspring) < self.population_size - self.elite_size:
                    random_config = self.search_space.sample_random_config()
                    offspring.append(Individual(random_config, generation + 1))

                self.population.replace_worst(offspring, self.elite_size)
                self.population.increment_generation()

            self._save_checkpoint(generation)

        total_time = time.time() - start_time

        best_individual = self.population.get_best(1)[0]

        result = {
            'best_individual': best_individual.to_dict(),
            'best_config': best_individual.config,
            'best_fitness': best_individual.fitness,
            'best_metrics': best_individual.metrics,
            'history': self.history,
            'total_time': total_time,
            'cache_stats': self.evaluator.get_cache_statistics()
        }

        if verbose:
            print(f"\n{'='*60}")
            print(f"Optimisation Complete!")
            print(f"{'='*60}")
            print(f"Total Time: {total_time:.2f}s")
            print(f"Best Fitness: {best_individual.fitness:.4f}")
            print(f"Best Config: {best_individual.config}")
            print(f"Best Metrics: {best_individual.metrics}")

        return result

    def _save_checkpoint(self, generation: int):
        checkpoint_file = os.path.join(self.checkpoint_dir, f"checkpoint_gen_{generation}.json")
        checkpoint = {
            'generation': generation,
            'population': [ind.to_dict() for ind in self.population.individuals],
            'history': self.history
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=_json_default)


from src.optimiser.individual import Individual
