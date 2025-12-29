#!/usr/bin/env python3
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.optimiser.evolution import EvolutionaryOptimiser
from src.tracking.logger import ExperimentLogger
from src.tracking.visualiser import Visualiser
from src.tracking.results import ResultsManager
from src.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evolutionary Hyperparameter Optimisation for CIFAR-10'
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to configuration YAML file'
    )

    parser.add_argument(
        '--experiment-name', '-e',
        type=str,
        default=None,
        help='Name of the experiment (overrides config)'
    )

    parser.add_argument(
        '--population-size', '-p',
        type=int,
        default=None,
        help='Population size (overrides config)'
    )

    parser.add_argument(
        '--generations', '-g',
        type=int,
        default=None,
        help='Number of generations (overrides config)'
    )

    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=None,
        help='Number of parallel workers (overrides config)'
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Output directory for results (overrides config)'
    )

    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable evaluation cache'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use for training'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)

    config = Config(args.config)

    if args.experiment_name:
        config.set('experiment', 'name', args.experiment_name)
    if args.population_size:
        config.set('evolution', 'population_size', args.population_size)
    if args.generations:
        config.set('evolution', 'num_generations', args.generations)
    if args.workers:
        config.set('parallel', 'num_workers', args.workers)
    if args.output_dir:
        config.set('experiment', 'output_dir', args.output_dir)
    if args.no_cache:
        config.set('experiment', 'use_cache', False)
    if args.device:
        config.set('training', 'device', args.device)

    experiment_name = config.get('experiment', 'name', 'cifar10_hpo')
    output_dir = config.get('experiment', 'output_dir', 'results')

    print("="*60)
    print("Evolutionary Hyperparameter Optimisation")
    print("="*60)
    print(f"Experiment: {experiment_name}")
    print(f"Population Size: {config.get('evolution', 'population_size')}")
    print(f"Generations: {config.get('evolution', 'num_generations')}")
    print(f"Workers: {config.get('parallel', 'num_workers') or 'auto'}")
    print(f"Device: {config.get('training', 'device')}")
    print(f"Output: {output_dir}/{experiment_name}")
    print("="*60)

    logger = ExperimentLogger(experiment_name, output_dir)

    optimiser = EvolutionaryOptimiser(
        population_size=config.get('evolution', 'population_size'),
        num_generations=config.get('evolution', 'num_generations'),
        crossover_rate=config.get('evolution', 'crossover_rate'),
        mutation_rate=config.get('evolution', 'mutation_rate'),
        elite_size=config.get('evolution', 'elite_size'),
        num_workers=config.get('parallel', 'num_workers'),
        data_dir=config.get('data', 'data_dir'),
        max_epochs=config.get('training', 'max_epochs'),
        device=config.get('training', 'device'),
        checkpoint_dir=os.path.join(logger.get_experiment_dir(), 'checkpoints')
    )

    result = optimiser.optimise(verbose=True)

    logger.save_final_results(result)
    logger.save_best_config(result['best_config'], result['best_metrics'])
    logger.export_to_csv(result['history'])

    results_manager = ResultsManager(logger.get_experiment_dir())
    report = results_manager.generate_report(result)
    print("\n" + report)

    visualiser = Visualiser(logger.get_experiment_dir())
    plots = visualiser.generate_all_plots(result)

    print(f"\nResults saved to: {logger.get_experiment_dir()}")
    print(f"Generated {len(plots)} visualisation plots")

    config.save(os.path.join(logger.get_experiment_dir(), 'config.yaml'))

    print("\nOptimisation complete!")


if __name__ == '__main__':
    main()
