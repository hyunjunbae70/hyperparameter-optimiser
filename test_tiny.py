#!/usr/bin/env python3
"""Tiny test - uses only 500 samples to verify everything works."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("1. Importing modules...")
import torch
from torch.utils.data import Subset
from src.data.cifar10_loader import CIFAR10DataLoader
from src.models.model_builder import ModelBuilder
from src.evaluation.trainer import ModelTrainer

print("2. Loading data (tiny subset)...")
data_loader = CIFAR10DataLoader(data_dir='./data')
data_loader.prepare_data()

# Get full loaders first, then subset
train_dataset = data_loader.train_dataset
val_dataset = data_loader.val_dataset

# Use only 500 training and 100 val samples
train_subset = Subset(train_dataset, range(500))
val_subset = Subset(val_dataset, range(100))

train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_subset, batch_size=64, shuffle=False, num_workers=0)

print(f"   Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

print("3. Building model...")
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

print("4. Training for 3 epochs...")
trainer = ModelTrainer(device='cpu')

model = model.to(trainer.device)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(3):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(trainer.device), targets.to(trainer.device)
        optimiser.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimiser.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(trainer.device), targets.to(trainer.device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100 * correct / total
    print(f"   Epoch {epoch+1}/3 - Loss: {total_loss/len(train_loader):.4f}, Val Acc: {acc:.1f}%")

print("5. SUCCESS! Everything works.")
print("\nYour system is fine - full training is just slow on CPU.")
print("For real experiments, use cloud/GPU or wait longer with quick_test.yaml")
