import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any
from src.models.cifar_net import ConfigurableCIFARNet


class ModelBuilder:
    @staticmethod
    def build_model(config: Dict[str, Any]) -> nn.Module:
        num_layers = int(config.get('num_layers', 3))
        base_channels = int(config.get('base_channels', 64))
        dropout_rate = float(config.get('dropout_rate', 0.2))

        model = ConfigurableCIFARNet(
            num_layers=num_layers,
            base_channels=base_channels,
            dropout_rate=dropout_rate,
            num_classes=10
        )

        return model

    @staticmethod
    def build_optimiser(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
        optimiser_name = config.get('optimiser', 'Adam')
        learning_rate = float(config.get('learning_rate', 0.001))
        weight_decay = float(config.get('weight_decay', 1e-4))
        momentum = float(config.get('momentum', 0.9))

        if optimiser_name == 'SGD':
            optimiser = optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay
            )
        elif optimiser_name == 'Adam':
            optimiser = optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimiser_name == 'AdamW':
            optimiser = optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimiser_name == 'RMSprop':
            optimiser = optim.RMSprop(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=momentum
            )
        else:
            raise ValueError(f"Unknown optimiser: {optimiser_name}")

        return optimiser

    @staticmethod
    def build_scheduler(optimiser: optim.Optimizer, config: Dict[str, Any]) -> optim.lr_scheduler._LRScheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimiser,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=False
        )
        return scheduler
