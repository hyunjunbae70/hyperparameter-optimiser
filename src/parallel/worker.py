import torch
from typing import Dict, Any
from src.models.model_builder import ModelBuilder
from src.data.cifar10_loader import CIFAR10DataLoader
from src.evaluation.trainer import ModelTrainer
from src.evaluation.fitness import MultiObjectiveFitness


def evaluate_individual_worker(config: Dict[str, Any], data_dir: str = './data',
                                max_epochs: int = 30, device: str = 'cpu',
                                verbose: bool = True) -> Dict[str, Any]:
    try:
        if verbose:
            print(f"  DEBUG: Worker started, loading data...", flush=True)
        torch.set_num_threads(1)

        data_loader = CIFAR10DataLoader(data_dir=data_dir)
        data_loader.prepare_data()

        batch_size = int(config.get('batch_size', 64))
        train_loader = data_loader.get_train_loader(batch_size=batch_size, num_workers=0)
        val_loader = data_loader.get_val_loader(batch_size=batch_size, num_workers=0)

        if verbose:
            print(f"  DEBUG: Data loaded, building model...", flush=True)

        model = ModelBuilder.build_model(config)
        optimiser = ModelBuilder.build_optimiser(model, config)
        scheduler = ModelBuilder.build_scheduler(optimiser, config)

        if verbose:
            print(f"  DEBUG: Model built, starting training...", flush=True)

        trainer = ModelTrainer(device=device)
        history = trainer.train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimiser=optimiser,
            scheduler=scheduler,
            num_epochs=max_epochs,
            early_stopping_patience=5,
            verbose=verbose
        )

        fitness_calculator = MultiObjectiveFitness()
        result = fitness_calculator.calculate_with_metrics(history)

        return {
            'success': True,
            'result': result,
            'error': None
        }

    except Exception as e:
        return {
            'success': False,
            'result': None,
            'error': str(e)
        }
