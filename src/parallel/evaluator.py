from multiprocessing import Pool, cpu_count
from typing import List, Dict, Any
from tqdm import tqdm
from src.optimiser.individual import Individual
from src.evaluation.cache import EvaluationCache
from src.parallel.worker import evaluate_individual_worker


class ParallelEvaluator:
    def __init__(self, num_workers: int = None, data_dir: str = './data',
                 max_epochs: int = 30, device: str = 'cpu',
                 use_cache: bool = True, cache_file: str = 'results/eval_cache.json'):
        if num_workers is None:
            self.num_workers = max(1, cpu_count() - 1)
        else:
            self.num_workers = num_workers

        self.data_dir = data_dir
        self.max_epochs = max_epochs
        self.device = device
        self.use_cache = use_cache

        if use_cache:
            self.cache = EvaluationCache(cache_file)
        else:
            self.cache = None

        self.cache_hits = 0
        self.cache_misses = 0

    def evaluate_population(self, population: List[Individual], verbose: bool = True) -> List[Individual]:
        unevaluated = [ind for ind in population if not ind.is_evaluated()]

        if not unevaluated:
            return population

        to_evaluate = []
        cached_results = []

        for ind in unevaluated:
            if self.use_cache and self.cache.has(ind.config_hash):
                cached_result = self.cache.get(ind.config_hash)
                ind.set_fitness(cached_result['fitness'], cached_result)
                cached_results.append(ind)
                self.cache_hits += 1
            else:
                to_evaluate.append(ind)
                self.cache_misses += 1

        if verbose and self.use_cache:
            print(f"Cache hits: {self.cache_hits}, Cache misses: {self.cache_misses}, "
                  f"Evaluating: {len(to_evaluate)}")

        if to_evaluate:
            configs = [ind.config for ind in to_evaluate]

            with Pool(processes=self.num_workers) as pool:
                args_list = [(config, self.data_dir, self.max_epochs, self.device)
                             for config in configs]

                if verbose:
                    results = []
                    with tqdm(total=len(args_list), desc="Evaluating") as pbar:
                        for result in pool.starmap(evaluate_individual_worker, args_list):
                            results.append(result)
                            pbar.update(1)
                else:
                    results = pool.starmap(evaluate_individual_worker, args_list)

            for ind, result in zip(to_evaluate, results):
                if result['success']:
                    ind.set_fitness(result['result']['fitness'], result['result'])

                    if self.use_cache:
                        self.cache.put(ind.config_hash, result['result'])
                else:
                    print(f"Error evaluating individual: {result['error']}")
                    ind.set_fitness(0.0, {'error': result['error']})

        return population

    def get_cache_statistics(self) -> Dict[str, Any]:
        stats = {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses)
                       if (self.cache_hits + self.cache_misses) > 0 else 0.0
        }

        if self.cache is not None:
            stats.update(self.cache.get_statistics())

        return stats
