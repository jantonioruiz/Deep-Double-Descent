import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms
import csv
import time

# Random seed "42"
np.random.seed(42)
torch.manual_seed(42)

num_classes = 10
img_rows, img_cols = 28, 28

def load_and_preprocess_mnist_data():
    """Carga los datos de MNIST y los preprocesa ajustando dimensiones y normalizando."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # [0, 1] check https://pytorch.org/vision/stable/transforms.html.
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    return train_dataset, test_dataset

def add_label_noise(dataset, noise_level=0.1):
    num_samples = len(dataset)
    num_noisy = int(noise_level * num_samples)
    
    indices = np.random.choice(num_samples, num_noisy, replace=False)
    
    for idx in indices:
        original_label = dataset.targets[idx]  
        
        noisy_label = original_label
        while noisy_label == original_label:
            noisy_label = np.random.randint(0, len(torch.unique(dataset.targets))) 
        
        dataset.targets[idx] = noisy_label 

def splitData(training_ds, test_ds, noise_level=0.1, add_noise=False):
    if add_noise:
        add_label_noise(training_ds, noise_level=noise_level)

    classes = training_ds.classes  
    batch_size = 256
    
    train_loader = DataLoader(training_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return classes, train_loader, test_loader

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)  # Conv1: 28x28x1 -> 28x28x6 (we use padding = 2 so the output of the first layer is 28x28 because our input images are 28x28)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # Conv2: 14x14x6 -> 10x10x16
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # Fully Connected Layer
        self.fc2 = nn.Linear(120, 84)  # Fully Connected Layer
        self.fc3 = nn.Linear(84, num_classes)  # Output Layer: 10 classes

    def num_flat_features(self, x):
        '''
        Get the number of features in a batch of tensors `x`.
        '''
        size = x.size()[1:]
        return np.prod(size)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))  # Conv1 + MaxPooling + ReLU (Max Pooling: 28x28x6 -> 14x14x6)
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))  # Conv2 + MaxPooling + ReLU (Max Pooling: 10x10x16 -> 5x5x16)
        x = x.view(-1, self.num_flat_features(x))  # Flatten: 5x5x16 -> 400
        x = F.relu(self.fc1(x))  # FC1 + ReLU
        x = F.relu(self.fc2(x))  # FC2 + ReLU
        x = self.fc3(x)  # Output Layer
        return x
    
def create_lenet5(num_classes):
    model = LeNet5(num_classes=num_classes)  
    return model

def train_and_evaluate_model(model, train_loader, test_loader, output_train_file, output_test_file, num_epochs=1000):
    """Entrena el modelo y evalúa el rendimiento en los conjuntos de entrenamiento y test."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    device = 'cuda'
    model = model.to(device)

    for epoch in range(num_epochs):
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        # Training
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            _, predicted = output.max(1)
            total_train += target.size(0)
            correct_train += predicted.eq(target).sum().item()

        train_loss = running_train_loss / len(train_loader)
        train_accuracy = correct_train / total_train

        # Evaluate
        model.eval()
        running_test_loss = 0.0
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                running_test_loss += loss.item()
                _, predicted = output.max(1)
                total_test += target.size(0)
                correct_test += predicted.eq(target).sum().item()

        test_loss = running_test_loss / len(test_loader)
        test_accuracy = correct_test / total_test

        with open(output_train_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, train_loss, train_accuracy])

        with open(output_test_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, test_loss, test_accuracy])

def saveHeader(output_train_file, output_test_file):
    with open(output_train_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['epoch', 'train_loss', 'train_accuracy'])

    with open(output_test_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['epoch', 'test_loss', 'test_accuracy'])

def main():
    noise = 0.1 # % of label noise
    training_ds, test_ds = load_and_preprocess_mnist_data()
    classes, train_loader, test_loader = splitData(training_ds, test_ds, noise_level=noise, add_noise=True)
    output_train_file = '/mnt/homeGPU/jaruiz/mnist_train4k_lenet5.txt'
    output_test_file = '/mnt/homeGPU/jaruiz/mnist_test4k_lenet5.txt'
    saveHeader(output_train_file, output_test_file)

    # Entrenamiento y evaluación con diferentes números de unidades
    num_classes = len(classes)
    num_epochs=1000

    # Tiempo de inicio 
    inicio = time.time()

    model = create_lenet5(num_classes)
    train_and_evaluate_model(model, train_loader, test_loader, output_train_file=output_train_file, output_test_file=output_test_file, num_epochs=num_epochs)

    # Tiempo de fin
    fin = time.time()

    # Tiempo total de ejecución
    tiempo_total = int(fin - inicio)
    horas = tiempo_total // 3600
    minutos = (tiempo_total % 3600) // 60
    segundos = tiempo_total % 60

    with open(output_train_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([])
        writer.writerow([f"La ejecución ha durado {horas} hora(s), {minutos} minuto(s) y {segundos} segundo(s)."])

if __name__ == "__main__":
    main()
