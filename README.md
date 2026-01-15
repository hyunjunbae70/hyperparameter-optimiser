# Hyperparameter Optimisation Engine

*Personal project demonstrating software engineering skills for internship applications. Built to show end-to-end system design, from evolutionary algorithms to cloud deployment.*

An evolutionary hyperparameter search system for PyTorch models using custom genetic operators, multi-objective fitness, and parallel evaluation. Optimises CIFAR-10 CNN architectures with mixed discrete-continuous hyperparameter spaces.

## Features

- **Evolutionary Search**: Custom genetic algorithm with population-based optimisation
- **Mixed Parameter Spaces**: Handles both continuous (learning rate, dropout) and discrete (optimiser, batch size, architecture) hyperparameters
- **Custom Operators**:
  - Gaussian mutation for continuous parameters with adaptive rates
  - Simulated Binary Crossover (SBX) for continuous values
  - Uniform crossover/mutation for discrete parameters
- **Multi-Objective Fitness**: Balances accuracy, training stability, and runtime efficiency
- **Early Stopping**: Patience-based early stopping to prevent overfitting
- **Evaluation Cache**: Hash-based caching to avoid redundant model training
- **Parallel Evaluation**: Asynchronous evaluation using Python multiprocessing
- **Experiment Tracking**: Custom JSON/CSV logging with matplotlib visualisations
- **Docker Support**: CPU-optimised container for deployment
- **Cloud Deployment**: Full Google Cloud Run integration with Cloud Storage

## Architecture

```
hyperparameter-optimiser/
├── src/
│   ├── optimiser/          # Evolutionary algorithm components
│   ├── search_space/       # Hyperparameter definitions
│   ├── evaluation/         # Fitness calculation & training
│   ├── models/             # Configurable CNN architecture
│   ├── parallel/           # Multiprocessing workers
│   ├── tracking/           # Logging & visualisation
│   └── data/               # CIFAR-10 data loading
├── configs/                # YAML configurations
├── deployment/             # Docker & Cloud deployment
└── scripts/                # Utility scripts
```

## Installation

### Local Installation

```bash
git clone https://github.com/hyunjunbae70/hyperparameter-optimiser.git
cd hyperparameter-optimiser
# inside venv (create, then 'source venv/bin/activate')
pip install -r requirements.txt

pip install -e .
```

### Docker Installation

```bash
docker build -t hyperparameter-optimiser -f deployment/Dockerfile .

docker run -v $(pwd)/results:/app/results hyperparameter-optimiser --generations 10 --population-size 20
```

## Quick Start

### Run Locally

```bash
python src/main.py --generations 10 --population-size 20

./scripts/run_local.sh my_experiment 10 20
```

### Using Configuration Files

```bash
python src/main.py --config configs/experiment_example.yaml
```

### CLI Options

```bash
python src/main.py \
  --config configs/default.yaml \
  --experiment-name my_experiment \
  --population-size 30 \
  --generations 20 \
  --workers 4 \
  --device cpu \
  --output-dir results \
  --seed 42
```

## Configuration

Configuration is managed via YAML files. See `configs/default.yaml` for all options:

```yaml
evolution:
  population_size: 30
  num_generations: 20
  crossover_rate: 0.8
  mutation_rate: 0.2
  elite_size: 3

training:
  max_epochs: 30
  early_stopping_patience: 5
  device: cpu

parallel:
  num_workers: null  # Auto-detect

fitness:
  w_accuracy: 1.0
  w_stability: 0.1
  w_runtime: 0.05
```

## Hyperparameter Search Space

The system optimises the following hyperparameters:

| Parameter | Type | Range/Choices |
|-----------|------|---------------|
| learning_rate | Continuous (log) | [1e-5, 1e-1] |
| optimiser | Categorical | SGD, Adam, AdamW, RMSprop |
| batch_size | Discrete | [32, 64, 128, 256] |
| dropout_rate | Continuous | [0.0, 0.5] |
| num_layers | Discrete | [2, 3, 4, 5] |
| base_channels | Discrete | [32, 64, 128] |
| weight_decay | Continuous (log) | [1e-6, 1e-2] |
| momentum | Continuous | [0.8, 0.99] |

## Multi-Objective Fitness Function

The fitness function balances three objectives:

```
fitness = w1 * accuracy - w2 * stability_penalty - w3 * runtime_penalty

where:
  - accuracy: Best validation accuracy (0-1)
  - stability_penalty: Std deviation of validation loss
  - runtime_penalty: Normalised training time
  - weights: w1=1.0, w2=0.1, w3=0.05
```

This encourages:
- High accuracy
- Stable training (low variance in validation loss)
- Efficient training (faster models preferred)

## Results & Visualisation

After optimisation, results are saved to `results/<experiment_name>/`:

```
results/my_experiment/
├── final_result.json       # Complete optimisation results
├── best_config.json        # Best hyperparameters found
├── config.yaml             # Experiment configuration
├── summary.csv             # Generation-wise metrics
├── report.txt              # Human-readable summary
├── fitness_evolution.png   # Fitness over generations
├── param_dist_*.png        # Parameter evolution plots
├── multi_objective_tradeoff.png  # Accuracy vs stability/runtime
└── checkpoints/            # Generation checkpoints
```

### Analyze Results

```bash
python scripts/analyze_results.py results/my_experiment

python scripts/analyze_results.py results/exp1 results/exp2 --compare
```

## Cloud Deployment (Google Cloud Run)

### Prerequisites

1. Google Cloud Project with billing enabled
2. Google Cloud SDK installed
3. Docker installed

### Deploy to Cloud Run

```bash
export PROJECT_ID=your-gcp-project-id
export GCS_BUCKET=your-gcs-bucket-name

gcloud config set project $PROJECT_ID

./scripts/deploy_cloudrun.sh $PROJECT_ID us-central1
```

### Configuration for Cloud

1. Update `deployment/cloudrun.yaml`:
   ```yaml
   env:
     - name: GCS_BUCKET
       value: "your-bucket-name"
     - name: GOOGLE_CLOUD_PROJECT
       value: "your-project-id"
   ```

2. Create a GCS bucket for results:
   ```bash
   gsutil mb gs://your-bucket-name
   ```

### Remote Execution

Results are automatically uploaded to Google Cloud Storage when running on Cloud Run.

## Design Goals

This project demonstrates:
- **Evolutionary algorithms**: Custom genetic operators for mixed parameter spaces
- **Parallel systems**: Multiprocessing with async evaluation and caching
- **Multi-objective optimisation**: Balancing accuracy, stability, and runtime
- **End-to-end deployment**: From local development to containerised cloud deployment
- **Clean architecture**: Modular design with proper separation of concerns

The system is designed to show how evolutionary search can explore hyperparameter spaces more efficiently than grid search, with proper caching and parallel evaluation reducing redundant computation.

## Development

### Running Tests

```bash
pip install -r requirements-dev.txt

pytest tests/ -v

pytest tests/ --cov=src --cov-report=html
```

### Code Quality

```bash
black src/ tests/

flake8 src/ tests/

mypy src/
```

## Technical Details

### Evolutionary Algorithm

- **Selection**: Tournament selection (k=3)
- **Crossover**: 80% rate, uniform for discrete, SBX for continuous
- **Mutation**: 20% base rate with adaptive decay
- **Elitism**: Top 10% preserved each generation
- **Population**: 20-50 individuals
- **Generations**: 10-30 (configurable)

### Parallel Evaluation

- Uses Python `multiprocessing.Pool`
- Async submission with Future objects
- Auto-detects CPU cores (or manual override)
- Fault-tolerant with retry logic
- Cache hits avoid redundant training

### Caching Strategy

- SHA-256 hash of hyperparameter configurations
- Persistent JSON cache across runs
- Deterministic training ensures validity
- Typical cache hit rate: 15-25%

## Troubleshooting

### Common Issues

**Out of Memory:**
```bash
python src/main.py --population-size 10 --workers 2
```

**Slow Execution:**
```bash
python src/main.py --max-epochs 15 --early-stopping-patience 3
```

**Docker Build Fails:**
Ensure you have permissions, are in the project root and Dockerfile path is correct:
```bash
docker build -t hpo -f deployment/Dockerfile .
```

## Learning Resources

Built while learning about:
- Genetic algorithms and evolutionary computation
- Multi-objective optimisation techniques
- PyTorch model training and hyperparameter tuning
- Docker containerisation and cloud deployment (GCP)
- Parallel programming with Python multiprocessing

## References

- CIFAR-10 dataset: https://www.cs.toronto.edu/~kriz/cifar.html
- PyTorch documentation: https://pytorch.org/docs/
- Deb & Agrawal (1995): Simulated Binary Crossover for continuous search spaces
