import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torch
import csv
import os

def loadDataset(dataset_name, data_augmentation=False):
    '''
    Loads the specified dataset ('MNIST', 'CIFAR10', or 'CIFAR100') with optional data augmentation.
    
    Args:
        dataset_name (str): Name of the dataset to load.
        data_augmentation (bool): Whether to apply data augmentation to the training data.
    '''

    if dataset_name == 'MNIST':
        transform_list = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))] # [0,1] check https://pytorch.org/vision/stable/transforms.html

    elif dataset_name == 'CIFAR10':
        transform_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    elif dataset_name == 'CIFAR100':
        transform_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported. Use 'MNIST', 'CIFAR10' or 'CIFAR100'.")

    # Apply data augmentation if specified
    if data_augmentation:
        augmentations = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()
        ]
        transform_list += augmentations 

    transform = transforms.Compose(transform_list)

    # Load datasets based on the specified name and transformations
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
    '''
    Randomly changes a percentage of labels to simulate uniform label noise.
    
    Args:
        dataset_name (str): Name of the dataset (must be one of 'MNIST', 'CIFAR10', 'CIFAR100').
        dataset (Subset): Subset to apply noise on.
        noise_level (float): Fraction of the dataset labels to randomly corrupt (value between 0 and 1).
    '''

    num_samples = len(dataset)
    num_noisy = int(noise_level * num_samples)
    
    indices = np.random.choice(num_samples, num_noisy, replace=False)

    # Apply noise by changing labels to random values, ensuring the new label is different from the original label
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
    '''
    Splits the dataset into smaller subsets, adds noise if specified and creates dataLoaders.
    
    Args:
        dataset_name (str): Name of the dataset.
        training_ds (Dataset): Training dataset.
        test_ds (Dataset): Test dataset.
        num_train_samples (int): Number of training samples to use.
        num_test_samples (int): Number of test samples to use.
        batch_size (int): Batch size for dataLoaders.
        noise_level (float): Fraction of training labels to corrupt.
    '''
    # Select random subsets for training and testing
    train_indices = np.random.choice(len(training_ds), num_train_samples, replace=False)
    test_indices = np.random.choice(len(test_ds), num_test_samples, replace=False)

    train_subset = Subset(training_ds, train_indices)
    test_subset = Subset(test_ds, test_indices)

    # Add noise to the training dataset if noise_level is greater than 0
    if noise_level > 0:
        if dataset_name == 'MNIST':
            addNoise('MNIST', train_subset, noise_level)
        elif dataset_name == 'CIFAR10':
            addNoise('CIFAR10', train_subset, noise_level)
        elif dataset_name == 'CIFAR100':
            addNoise('CIFAR100', train_subset, noise_level)
        else:
            raise ValueError(f"Dataset '{dataset_name}' not supported. Use 'MNIST', 'CIFAR10' or 'CIFAR100'.")

    # Get the class labels from the dataset
    if dataset_name == 'MNIST' or dataset_name == 'CIFAR10' or dataset_name == 'CIFAR100':
        classes = training_ds.classes
    else:
        raise ValueError("Error while obtaining the number of classes from the dataset.")

    # Create DataLoader for training and testing subsets
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    return classes, train_loader, test_loader

def saveHeader(output_train_file, output_test_file, model_name, **params):
    '''
    Writes the header and model information to the specified output files.
    
    Args:
        output_train_file (str): Path to the training results file.
        output_test_file (str): Path to the testing results file.
        model_name (str): Name of the model.
        **params: Any additional model parameters to save.
    '''

    param_str = " | ".join(f"{k}: {v}" for k, v in params.items())
    model_info = f"│  Model: {model_name}  |  {param_str}  │"

    border = "─" * (len(model_info) - 2)
    header_box = f"┌{border}┐\n{model_info}\n└{border}┘"

    def write_header(file_path, header_box, header_row):
        ''' 
        Checks if the specified file exists and is empty. If the file doesn't exist,  it creates the file and writes the header information.

        Args:
            file_path (str): The path where the file will be saved.
            header_box (str): A formatted string containing the model details to be displayed as a box.
            header_row (list): The column names for the file.
        '''
        if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
            with open(file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([header_box]) # Write model information as a box
                writer.writerow([]) # Empty row
                writer.writerow(header_row) # Write the column headers

    # Define headers for training and testing results
    train_header = ['units', 'epochs', 'train_loss', 'train_accuracy', 'train_error']
    test_header = ['units', 'epochs', 'test_loss', 'test_accuracy', 'test_error']

    # Write headers to respective files
    write_header(output_train_file, header_box, train_header)
    write_header(output_test_file, header_box, test_header)

def train_and_evaluate_model(model, criterion, optimizer, train_loader, test_loader, model_dimension, num_epochs, output_train_file, output_test_file):
    '''
    Trains the model and evaluates it on the test set at each epoch, saving the training and testing metrics to the corresponding files after every epoch.
    
    Args:
        model: The neural network model.
        criterion: The loss function.
        optimizer: Optimizer for training.
        train_loader (DataLoader): DataLoader for the training set.
        test_loader (DataLoader): DataLoader for the test set.
        model_dimension (int): Model size.
        num_epochs (int): Number of training epochs.
        output_train_file (str): Path to the training file.
        output_test_file (str): Path to the testing file.
    '''

    # Set the model to use GPU only
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