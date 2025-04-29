import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torch
import csv
import os

def loadDataset(dataset_name, data_augmentation=False):
    if dataset_name == 'MNIST':
        transform_list = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))] # [0,1] check https://pytorch.org/vision/stable/transforms.html

    elif dataset_name == 'CIFAR10':
        transform_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    elif dataset_name == 'CIFAR100':
        transform_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported. Use 'MNIST', 'CIFAR10' or 'CIFAR100'.")

    if data_augmentation:
        augmentations = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()
        ]
        transform_list += augmentations 

    transform = transforms.Compose(transform_list)

    if dataset_name == 'MNIST':
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    elif dataset_name == 'CIFAR10':
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    elif dataset_name == 'CIFAR100':
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    return train_dataset, test_dataset

def addNoise(dataset_name, dataset, noise_level):
    num_samples = len(dataset)
    num_noisy = int(noise_level * num_samples)
    
    indices = np.random.choice(num_samples, num_noisy, replace=False)

    if dataset_name == 'MNIST' or dataset_name == 'CIFAR10' or dataset_name == 'CIFAR100':
        for idx in indices:
            original_index = dataset.indices[idx]
            original_label = dataset.dataset.targets[original_index]

            noisy_label = original_label
            while noisy_label == original_label:
                noisy_label = np.random.randint(0, len(dataset.dataset.classes))

            dataset.dataset.targets[original_index] = noisy_label
    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported. Use 'MNIST', 'CIFAR10' or 'CIFAR100'.")

def splitData(dataset_name, training_ds, test_ds, num_train_samples=4000, num_test_samples=1000, batch_size=128, noise_level=0.1):
    train_indices = np.random.choice(len(training_ds), num_train_samples, replace=False)
    test_indices = np.random.choice(len(test_ds), num_test_samples, replace=False)

    train_subset = Subset(training_ds, train_indices)
    test_subset = Subset(test_ds, test_indices)

    if noise_level > 0:
        if dataset_name == 'MNIST':
            addNoise('MNIST', train_subset, noise_level)
        elif dataset_name == 'CIFAR10':
            addNoise('CIFAR10', train_subset, noise_level)
        elif dataset_name == 'CIFAR100':
            addNoise('CIFAR100', train_subset, noise_level)
        else:
            raise ValueError(f"Dataset '{dataset_name}' not supported. Use 'MNIST', 'CIFAR10' or 'CIFAR100'.")

    if dataset_name == 'MNIST' or dataset_name == 'CIFAR10' or dataset_name == 'CIFAR100':
        classes = training_ds.classes
    else:
        raise ValueError("Error while obtaining the number of classes from the dataset.")

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    return classes, train_loader, test_loader

def saveHeader(output_train_file, output_test_file, model_name, **params):
    param_str = " | ".join(f"{k}: {v}" for k, v in params.items())
    model_info = f"│  Model: {model_name}  |  {param_str}  │"

    border = "─" * (len(model_info) - 2)
    header_box = f"┌{border}┐\n{model_info}\n└{border}┘"

    def write_header(file_path, header_box, header_row):
        if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
            with open(file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([header_box])
                writer.writerow([]) 
                writer.writerow(header_row)

    train_header = ['units', 'epochs', 'train_loss', 'train_accuracy', 'train_error']
    test_header = ['units', 'epochs', 'test_loss', 'test_accuracy', 'test_error']

    write_header(output_train_file, header_box, train_header)
    write_header(output_test_file, header_box, test_header)

def train_and_evaluate_model(model, criterion, optimizer, train_loader, test_loader, model_dimension, num_epochs, output_train_file, output_test_file):
    # GPU only
    device = 'cuda'
    model = model.to(device)

    for epoch in range(num_epochs):
        # TRAINING
        # Model to training mode
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the gradients of all model parameters to prevent accumulation
            optimizer.zero_grad()  

            # Forward pass: compute predictions from inputs
            outputs = model(inputs)  
            # Compute the loss 
            loss = criterion(outputs, labels) 

            # Backpropagation 
            loss.backward()  
            # Update model parameters based on gradients
            optimizer.step()  

            # Accumulate the batch loss to calculate the epoch's total loss
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Train metrics
        train_loss = round(running_loss / len(train_loader), 6)
        train_accuracy = round(correct / total, 6)
        train_error = 1 - train_accuracy

        # Save train results each epoch
        with open(output_train_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([model_dimension, epoch + 1, train_loss, train_accuracy, train_error])

        # EVALUATE
        # Model to evaluation mode
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
 
        # Disabling gradient calculation for efficiency, as we don’t need to update parameters while evaluating
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                loss = criterion(outputs, labels)
                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Test metrics
        test_loss = round(running_loss / len(test_loader), 6)
        test_accuracy = round(correct / total, 6)
        test_error = 1 - test_accuracy

        # Save test results each epoch
        with open(output_test_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([model_dimension, epoch + 1, test_loss, test_accuracy, test_error])