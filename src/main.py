import numpy as np
import torch.nn as nn
import torch.optim as optim
import argparse
import csv
import time
from models import createNet
from utils import loadDataset, splitData, saveHeader, train_and_evaluate_model

# Main function
def main(args):
    # Convert arguments to a dictionary
    args_dict = vars(args)

    # Process and parse command-line arguments 
    dataset = args_dict["dataset"]
    model = args_dict["model"]
    output_train_file = args_dict["output_train"]
    output_test_file = args_dict["output_test"]
    data_augmentation = args_dict["data_augmentation"]
    num_train_samples = args_dict["num_train_samples"]
    num_test_samples = args_dict["num_test_samples"]
    noise = args_dict["noise"]
    num_epochs = args_dict["epochs"]
    batch_size = args_dict["batch_size"]
    # criterion_name = args_dict["criterion"]  # Variable not used (we only use CrossEntropyLoss)
    # optimizer_name = args_dict["optimizer"]  # Variable not used (we only use Adam optimizer)
    learning_rate = args_dict["learning_rate"]

    units = args_dict["units"]
    units_range = args_dict["units_range"]
    # If units_range is defined, extract the min and max values
    if units_range:
        units_min, units_max = units_range  
    else:
        units_min, units_max = None, None  

    # Load and preprocess dataset
    training_ds, test_ds = loadDataset(dataset)

    # Split data into training and testing sets
    classes, train_loader, test_loader = splitData(dataset_name=dataset, 
                                                   training_ds=training_ds, 
                                                   test_ds=test_ds, 
                                                   num_train_samples=num_train_samples, 
                                                   num_test_samples=num_test_samples, 
                                                   batch_size=batch_size, 
                                                   noise_level=noise)
    
    # Save information about the experiment
    saveHeader(output_train_file, output_test_file, 
               model, 
               dataset=f"{dataset}({num_train_samples}/{num_test_samples})", 
               data_augmentation=data_augmentation, 
               noise=noise, 
               batch_size=batch_size, 
               learning_rate=learning_rate, 
               epochs=num_epochs)

    # Total number of classes of the chosen dataset
    num_classes = len(classes)

    # If 'units' is specified, use that value; otherwise, create a range from 'units_min' to 'units_max'
    # For the 'ResNet18' model, set the number of units to 64 (default ResNet18 architecture)
    if model == 'ResNet18':
        units = 64
    else:
        if units is not None:
            num_units = [units] 
        else:
            num_units = np.arange(units_min, units_max + 1) 

    # Set input size and number of channels based on the chosen dataset
    if dataset == 'MNIST':
        input_size = 28*28 
        in_channels = 1 # Grayscale images
    else:
        input_size = 32*32 # For CIFAR10 & CIFAR100 datasets
        in_channels = 3 # RGB images

    # Start time of the execution
    start_time = time.time()

    # Loop over the units, create and train the corresponding neural network
    for u in num_units:
        # Create the corresponding architecture
        if model == 'TwoLayerNN' or model == 'DeepNN':
            net = createNet(model, input_size=input_size, size=u, num_classes=num_classes)
    
        elif model == 'ThreeLayerCNN' or model == 'DeepCNN' or model == 'PreActResNet':
            net = createNet(model, in_channels=in_channels, k=u, num_classes=num_classes)
        
        else:
            net = createNet(model, num_classes)

        # Set up the loss function
        criterion = nn.CrossEntropyLoss()

        # Set up the optimizer with the specified learning rate
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)

        # Train and evaluate the model with the specified parameters and store the results
        train_and_evaluate_model(model=net, 
                                criterion=criterion,
                                optimizer=optimizer, 
                                train_loader=train_loader, 
                                test_loader=test_loader, 
                                model_dimension=u, 
                                num_epochs=num_epochs, 
                                output_train_file=output_train_file, 
                                output_test_file=output_test_file)

    # End time of the execution
    final_time = time.time()

    # Calculate the total execution time
    total_time = int(final_time - start_time)
    hours = total_time // 3600
    minutes = (total_time % 3600) // 60
    seconds = total_time % 60

    # Store the execution time
    with open(output_train_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([])
        writer.writerow([f"Total execution time: {hours} hour(s) {minutes} minute(s) and {seconds} second(s)."])

# Main block
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train different neural network models on various datasets with customizable hyperparameters.")

    # Required arguments
    parser.add_argument("--dataset", type=str, required=True, choices=["MNIST", "CIFAR10", "CIFAR100"],
                        help="Dataset to be used for training (Options: MNIST, CIFAR10, CIFAR100, Flowers102)")
    
    parser.add_argument("--model", type=str, required=True, choices=["TwoLayerNN", "DeepNN", "ThreeLayerCNN", "DeepCNN", "PreActResNet", "ResNet18"],
                        help="Neural network architecture to train (Options: TwoLayerNN, DeepNN, ThreeLayerCNN, DeepCNN, PreActResNet, ResNet18)")
    
    parser.add_argument("--output_train", type=str, required=True,
                        help="File to save training results")
    
    parser.add_argument("--output_test", type=str, required=True,
                        help="File to save testing results")
    
    # Mutually exclusive arguments: either --units or --units_range is required
    units_group = parser.add_mutually_exclusive_group(required=True)
    units_group.add_argument("--units", type=int,
                             help="Number of units for a single model")
    
    units_group.add_argument("--units_range", type=int, nargs=2, metavar=("UNITS_MIN", "UNITS_MAX"),
                             help="Range of units to use (Specify two integers: min and max, e.g., --units_range 32 64)")


    # Optional arguments
    parser.add_argument("--data_augmentation", action="store_true",
                    help="Enable data augmentation (Default: False)")
    
    parser.add_argument("--num_train_samples", type=int, default=4000,
                        help="Number of training samples to use (Default: 4000)")
    
    parser.add_argument("--num_test_samples", type=int, default=1000,
                        help="Number of test samples to use (Default: 1000)")
    
    parser.add_argument("--noise", type=float, default=0.1,
                        help="Percentage of label noise (Default: 0.1, meaning 10%% of labels are randomly altered)")
    
    parser.add_argument("--epochs", type=int, default=1000,
                        help="Total number of training epochs (Default: 1000)")
    
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for training and evaluation (Default: 128)")
    
    parser.add_argument("--criterion", type=str, default="CrossEntropyLoss",
                        help="Loss function to optimize the model (Default: CrossEntropyLoss)")
    
    parser.add_argument("--optimizer", type=str, default="Adam",
                        help="Optimizer for training (Default: Adam)")
    
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate for the optimizer (Default: 0.001)")

    # Parse the arguments
    args = parser.parse_args()

    # Validation for units_range
    if args.units_range is not None:
        units_min, units_max = args.units_range
        if units_min > units_max:
            parser.error("Invalid --units_range: UNITS_MIN must be less than or equal to UNITS_MAX")

    # Validation for noise
    if args.noise < 0:
        parser.error("Invalid --noise: Noise level must be positive")

    # Call the main function
    main(args)
