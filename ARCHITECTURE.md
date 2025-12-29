# Architecture Documentation

## System Overview

The Hyperparameter Optimisation Engine is designed as a modular evolutionary search system with the following key design principles:

1. **Separation of Concerns**: Each module handles a specific responsibility
2. **Extensibility**: Easy to add new operators, fitness functions, or search spaces
3. **Parallelism**: Built-in support for concurrent evaluation
4. **Reproducibility**: Deterministic execution with configurable random seeds

## Core Components

### 1. Search Space (`src/search_space/`)

**Purpose**: Define the hyperparameter space to be optimised

**Design Decisions**:
- Unified interface for continuous, discrete, and categorical parameters
- Log-scale support for parameters spanning multiple orders of magnitude
- Built-in validation and clipping to ensure valid configurations

**Key Classes**:
- `HyperparameterSpace`: Encapsulates all parameter definitions
- `ParamType`: Enum for parameter types

### 2. Individual Representation (`src/optimiser/individual.py`)

**Purpose**: Represent a single candidate solution

**Design Decisions**:
- Immutable config with copy-on-modify semantics
- SHA-256 hashing for efficient caching and duplicate detection
- Lazy hash computation for performance

**Key Features**:
- Configuration serialization (to/from dict)
- Fitness storage with associated metrics
- Generation tracking for analysis

### 3. Genetic Operators (`src/optimiser/operators.py`)

**Purpose**: Implement variation operators for evolution

**Design Decisions**:
- **Mutation**:
  - Gaussian mutation for continuous parameters with adaptive sigma
  - Uniform random for discrete/categorical
  - Generation-based decay to balance exploration/exploitation
- **Crossover**:
  - Simulated Binary Crossover (SBX) for continuous (preserves distribution)
  - Uniform crossover for discrete (fair mixing)
- **Selection**:
  - Tournament selection (k=3) for diversity
  - Elitism to preserve best solutions

**Adaptive Mechanisms**:
```python
mutation_rate(t) = base_rate * (1 - 0.5 * t/T)
sigma(t) = range * 0.1 / (1 + 0.01 * t)
```

### 4. Population Management (`src/optimiser/population.py`)

**Purpose**: Manage the collection of individuals

**Design Decisions**:
- Generational replacement with elitism
- Lazy evaluation (individuals don't auto-evaluate)
- Statistics tracking for monitoring

**Operations**:
- Initialization (random or with seed individual)
- Selection of best/worst
- Replacement strategies

### 5. Fitness Evaluation (`src/evaluation/`)

**Purpose**: Train models and compute multi-objective fitness

**Components**:
- **Trainer**: Early-stopping enabled PyTorch training
- **Fitness Calculator**: Multi-objective weighted sum
- **Metrics**: Accuracy, stability, runtime tracking
- **Cache**: Hash-based persistent cache

**Multi-Objective Formula**:
```
fitness = w1*accuracy - w2*stability - w3*runtime
```

**Design Rationale**:
- Weighted sum is simple and efficient
- Penalties normalised to [0,1] range
- Weights tuned empirically (w1=1.0, w2=0.1, w3=0.05)

### 6. Model Architecture (`src/models/`)

**Purpose**: Define trainable CNN for CIFAR-10

**Design Decisions**:
- Configurable depth (num_layers)
- Configurable width (base_channels)
- Batch normalization for stability
- Adaptive pooling for variable architectures
- Dropout for regularization

**Architecture Pattern**:
```
[Conv-BN-ReLU-Conv-BN-ReLU-MaxPool] x num_layers
AdaptiveAvgPool
FC-BN-ReLU-Dropout-FC
```

### 7. Parallel Evaluation (`src/parallel/`)

**Purpose**: Accelerate fitness computation

**Design Decisions**:
- Process-based parallelism (vs threading) for CPU-bound work
- Each worker is isolated (no shared state)
- Async submission with progress tracking
- Fault tolerance with retry logic

**Worker Design**:
```python
def worker(config):
    # Isolated process
    torch.set_num_threads(1)  # Prevent over-subscription
    train_model(config)
    return fitness
```

**Concurrency Pattern**:
```
Pool(n_workers)
  └─> [Worker1, Worker2, ..., WorkerN]
       Each trains 1 model at a time
```

### 8. Experiment Tracking (`src/tracking/`)

**Purpose**: Log, visualise, and persist results

**Components**:
- **Logger**: JSONL + CSV output
- **Visualiser**: matplotlib plots
- **Results Manager**: Checkpoint management
- **GCS Uploader**: Cloud storage integration

**Logging Strategy**:
- JSONL for append-only, parseable logs
- CSV for easy import to analysis tools
- JSON for structured results
- PNG for visualisations

### 9. Evolution Engine (`src/optimiser/evolution.py`)

**Purpose**: Orchestrate the complete optimisation loop

**Algorithm**:
```
Initialize population
For each generation:
    Evaluate unevaluated individuals (parallel)
    Log statistics
    Select parents (tournament)
    Create offspring (crossover + mutation)
    Replace worst, keep elite
    Checkpoint
Return best individual
```

**Design Decisions**:
- Generational model (all offspring created before replacement)
- Elitism ensures monotonic improvement
- Checkpointing for fault tolerance and analysis

## Data Flow

```
User Config (YAML)
  ↓
EvolutionaryOptimiser
  ↓
Population.initialize()
  ↓
┌─────────────────────────┐
│  Generation Loop        │
│  ┌──────────────────┐  │
│  │ ParallelEvaluator│  │
│  │   ├─> Worker 1   │  │
│  │   ├─> Worker 2   │  │
│  │   └─> Worker N   │  │
│  └──────────────────┘  │
│         ↓               │
│  Update Population      │
│         ↓               │
│  GeneticOperators       │
│  (select, cross, mutate)│
│         ↓               │
│  Logger/Visualiser      │
└─────────────────────────┘
  ↓
Results (JSON/CSV/PNG)
```

## Key Design Patterns

### 1. Strategy Pattern
- Different mutation strategies for different parameter types
- Pluggable fitness functions
- Configurable selection mechanisms

### 2. Builder Pattern
- `ModelBuilder`: Constructs models from configs
- `Config`: Builds runtime config from YAML + overrides

### 3. Template Method
- `Trainer.train()`: Defines training loop structure
- Subclass can override epoch/validation logic

### 4. Observer Pattern
- Logger observes generation events
- Visualiser updates plots on events

## Performance Optimisations

### 1. Caching
- **What**: Fitness evaluation results
- **Why**: Genetic operators can produce duplicates
- **Impact**: 15-25% cache hit rate → 15-25% speedup

### 2. Multiprocessing
- **What**: Parallel model training
- **Why**: CPU-bound training workload
- **Impact**: Near-linear speedup (4 workers → ~3.5x faster)

### 3. Early Stopping
- **What**: Stop training when validation loss plateaus
- **Why**: Avoid wasted epochs
- **Impact**: 30-40% reduction in training time

### 4. Lazy Evaluation
- **What**: Only compute fitness when needed
- **Why**: Elite individuals don't need re-evaluation
- **Impact**: Avoid 10% of evaluations

## Scalability Considerations

### Current Limits
- Population size: 50 (beyond this, evaluation time dominates)
- Workers: CPU cores (I/O not bottleneck)
- Generations: 30 (diminishing returns after)

### Bottlenecks
1. **Model training**: Dominates runtime (90%+)
2. **CIFAR-10 data loading**: Minor (< 5%)
3. **Genetic operations**: Negligible (< 1%)

### Future Scaling
- **Distributed evaluation**: Use Ray/Dask for multi-node
- **GPU support**: Modify worker to use CUDA
- **Larger datasets**: Stream data, don't preload

## Configuration System

### YAML-Based Config
- Hierarchical structure (evolution, training, parallel, etc.)
- Default values with user overrides
- CLI arguments take precedence

### Override Priority
```
CLI args > Config file > Defaults
```

### Validation
- Type checking on load
- Range validation for numeric params
- Required fields enforcement

## Testing Strategy

### Unit Tests
- Operator correctness (mutation, crossover)
- Fitness calculation accuracy
- Population management logic

### Integration Tests
- End-to-end evolution run (1 generation, small pop)
- Config loading and merging
- Cache persistence

### Manual Testing
- Full optimisation runs
- Docker build and execution
- Cloud deployment

## Deployment Architecture

### Local
```
Python 3.10 + PyTorch (CPU)
  └─> Multiprocessing workers
       └─> CIFAR-10 from local disk
```

### Docker
```
python:3.10-slim base image
  └─> Multi-stage build (reduce size)
       └─> CPU-only PyTorch
            └─> Volume mount for results
```

### Cloud Run
```
GCR image
  └─> 4 vCPU, 8GB RAM
       └─> Cloud Storage for results
            └─> Cloud Logging for monitoring
```

## Extensibility Points

To extend the system:

1. **New search space**: Edit `HyperparameterSpace`
2. **New operators**: Subclass `GeneticOperators`
3. **New fitness**: Implement `FitnessCalculator` interface
4. **New model**: Implement in `models/`, update `ModelBuilder`
5. **New dataset**: Implement `DataLoader` interface

## References

- Deb & Agrawal (1995): Simulated Binary Crossover
- Eiben & Smith (2015): Evolutionary Computing
- Bergstra & Bengio (2012): Random Search vs Grid Search
