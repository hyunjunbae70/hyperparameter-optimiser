import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Tuple
import time
import numpy as np


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.should_stop


class ModelTrainer:
    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)

    def train_epoch(self, model: nn.Module, train_loader: DataLoader,
                    optimiser: torch.optim.Optimizer, criterion: nn.Module) -> Tuple[float, float]:
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            optimiser.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimiser.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        return epoch_loss, epoch_acc

    def validate(self, model: nn.Module, val_loader: DataLoader,
                 criterion: nn.Module) -> Tuple[float, float]:
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        val_loss = running_loss / total
        val_acc = correct / total

        return val_loss, val_acc

    def train(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
              optimiser: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler,
              num_epochs: int = 30, early_stopping_patience: int = 5) -> Dict[str, Any]:
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()

        early_stopping = EarlyStopping(patience=early_stopping_patience)

        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epochs_trained': 0
        }

        start_time = time.time()

        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(model, train_loader, optimiser, criterion)

            val_loss, val_acc = self.validate(model, val_loader, criterion)

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['epochs_trained'] = epoch + 1

            if early_stopping(val_loss):
                break

        training_time = time.time() - start_time
        history['training_time'] = training_time

        history['best_val_acc'] = max(history['val_acc'])
        history['best_val_loss'] = min(history['val_loss'])
        history['final_val_acc'] = history['val_acc'][-1]
        history['final_val_loss'] = history['val_loss'][-1]

        history['val_loss_std'] = np.std(history['val_loss'])

        return history
