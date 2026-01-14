#!/usr/bin/env python3
"""Recover final results from checkpoint file."""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.optimiser.individual import _json_default

# Load checkpoint
checkpoint_file = "results/cifar10_hpo/checkpoints/checkpoint_gen_1.json"
with open(checkpoint_file, 'r') as f:
    checkpoint = json.load(f)

# Find best individual by fitness
population = checkpoint['population']
best_ind = max(population, key=lambda x: x['fitness'])

# Build final result
result = {
    'best_individual': best_ind,
    'best_config': best_ind['config'],
    'best_fitness': best_ind['fitness'],
    'best_metrics': best_ind['metrics'],
    'history': checkpoint['history'],
    'total_time': sum(ind['metrics']['training_time'] for ind in population),
    'cache_stats': {
        'cache_hits': 0,
        'cache_misses': len(population),
        'hit_rate': 0.0
    }
}

# Save final_result.json
with open("results/cifar10_hpo/final_result.json", 'w') as f:
    json.dump(result, f, indent=2, default=_json_default)
print("Saved: final_result.json")

# Save best_config.json
from datetime import datetime
best_config_data = {
    'config': best_ind['config'],
    'metrics': best_ind['metrics'],
    'timestamp': datetime.now().isoformat()
}
with open("results/cifar10_hpo/best_config.json", 'w') as f:
    json.dump(best_config_data, f, indent=2, default=_json_default)
print("Saved: best_config.json")

# Generate report
report_lines = [
    "=" * 60,
    "Hyperparameter Optimisation Results",
    "=" * 60,
    "",
    f"Best Fitness: {best_ind['fitness']:.4f}",
    "",
    "Best Metrics:",
]
for key, value in best_ind['metrics'].items():
    if isinstance(value, float):
        report_lines.append(f"  {key}: {value:.4f}")
    else:
        report_lines.append(f"  {key}: {value}")

report_lines.append("\nBest Configuration:")
for key, value in best_ind['config'].items():
    report_lines.append(f"  {key}: {value}")

report_lines.append(f"\nTotal Optimisation Time: {result['total_time']:.2f}s")
report_lines.append("=" * 60)

report_text = "\n".join(report_lines)
with open("results/cifar10_hpo/report.txt", 'w') as f:
    f.write(report_text)
print("Saved: report.txt")

# Save summary.csv
import csv
with open("results/cifar10_hpo/summary.csv", 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Generation', 'Best Fitness', 'Mean Fitness'])
    history = checkpoint['history']
    for i in range(len(history['generations'])):
        writer.writerow([
            history['generations'][i],
            history['best_fitness'][i],
            history['mean_fitness'][i]
        ])
print("Saved: summary.csv")

print("\nRegenerating plots...")
from src.tracking.visualiser import Visualiser

visualiser = Visualiser("results/cifar10_hpo")
plots = visualiser.generate_all_plots(result)
print(f"Generated {len(plots)} plots")

print("\nRecovery complete!")
print(f"\nBest config found: {best_ind['fitness']:.4f} fitness, {best_ind['metrics']['accuracy']*100:.1f}% accuracy")
