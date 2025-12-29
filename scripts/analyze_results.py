#!/usr/bin/env python3
import argparse
import json
import os
import sys


def load_results(experiment_dir):
    final_result_path = os.path.join(experiment_dir, 'final_result.json')
    if not os.path.exists(final_result_path):
        print(f"Error: No results found in {experiment_dir}")
        return None

    with open(final_result_path, 'r') as f:
        return json.load(f)


def print_summary(result):
    print("="*60)
    print("Experiment Summary")
    print("="*60)
    print(f"Best Fitness: {result['best_fitness']:.4f}")
    print(f"Total Time: {result['total_time']:.2f}s")
    print()

    print("Best Configuration:")
    for key, value in result['best_config'].items():
        print(f"  {key}: {value}")
    print()

    print("Best Metrics:")
    for key, value in result['best_metrics'].items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print()

    if 'cache_stats' in result:
        print("Cache Statistics:")
        for key, value in result['cache_stats'].items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")


def compare_experiments(exp_dirs):
    results = []
    for exp_dir in exp_dirs:
        result = load_results(exp_dir)
        if result:
            results.append((os.path.basename(exp_dir), result))

    if not results:
        print("No valid experiments found")
        return

    print("="*60)
    print("Experiment Comparison")
    print("="*60)
    print(f"{'Experiment':<20} {'Best Fitness':<15} {'Total Time':<12}")
    print("-"*60)

    for name, result in results:
        print(f"{name:<20} {result['best_fitness']:<15.4f} {result['total_time']:<12.2f}s")


def main():
    parser = argparse.ArgumentParser(description='Analyze HPO experiment results')
    parser.add_argument('experiment_dirs', nargs='+', help='Experiment directory/directories')
    parser.add_argument('--compare', action='store_true', help='Compare multiple experiments')

    args = parser.parse_args()

    if args.compare and len(args.experiment_dirs) > 1:
        compare_experiments(args.experiment_dirs)
    else:
        result = load_results(args.experiment_dirs[0])
        if result:
            print_summary(result)


if __name__ == '__main__':
    main()
