#!/usr/bin/env python3
"""Quick test to isolate where the hang occurs."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("1. Importing modules...")
import torch
from src.data.cifar10_loader import CIFAR10DataLoader
from src.models.model_builder import ModelBuilder
from src.evaluation.trainer import ModelTrainer

print("2. Loading data...")
data_loader = CIFAR10DataLoader(data_dir='./data')
data_loader.prepare_data()

print("3. Creating data loaders...")
train_loader = data_loader.get_train_loader(batch_size=64, num_workers=0)
val_loader = data_loader.get_val_loader(batch_size=64, num_workers=0)

print("4. Building model...")
config = {
    'learning_rate': 0.01,
    'optimiser': 'Adam',
    'batch_size': 64,
    'dropout_rate': 0.2,
    'num_layers': 2,
    'base_channels': 32,
    'weight_decay': 1e-4,
    'momentum': 0.9
}
model = ModelBuilder.build_model(config)
optimiser = ModelBuilder.build_optimiser(model, config)
scheduler = ModelBuilder.build_scheduler(optimiser, config)

print("5. Training for 1 epoch...")
trainer = ModelTrainer(device='cpu')

model = model.to(trainer.device)
criterion = torch.nn.CrossEntropyLoss()

model.train()
for i, (inputs, targets) in enumerate(train_loader):
    inputs, targets = inputs.to(trainer.device), targets.to(trainer.device)
    optimiser.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimiser.step()

    if i % 100 == 0:
        print(f"   Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}")

print("6. Done!")
