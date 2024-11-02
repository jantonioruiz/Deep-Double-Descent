import numpy as np
import torch
import torch.nn as nn
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
        original_index = dataset.indices[idx]  
        original_label = dataset.dataset.targets[original_index]  
        
        noisy_label = original_label
        while noisy_label == original_label:
            noisy_label = np.random.randint(0, len(dataset.dataset.classes))
        
        dataset.dataset.targets[original_index] = noisy_label

def splitData(training_ds, test_ds, noise_level=0.1, add_noise=False):
    train_indices = np.random.choice(len(training_ds), 4000, replace=False)  
    test_indices = np.random.choice(len(test_ds), 1000, replace=False) 
    
    train_subset = Subset(training_ds, train_indices)
    test_subset = Subset(test_ds, test_indices)

    if add_noise:
        add_label_noise(train_subset, noise_level=noise_level) 

    classes = training_ds.classes  
    batch_size = 256
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    return classes, train_loader, test_loader

class SimpleNN(nn.Module):
    def __init__(self, size=1, num_classes=10):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()  
        self.fc1 = nn.Linear(28 * 28, size)  
        self.fc2 = nn.Linear(size, num_classes) 
    
    def forward(self, x):
        x = self.flatten(x)  
        x = torch.relu(self.fc1(x))  
        x = self.fc2(x)  
        return x
    
def create_2nn(size, num_classes):
    model = SimpleNN(size=size, num_classes=num_classes)  
    return model

def train_and_evaluate_model(model, train_loader, test_loader, num_epochs=1000):
    """Entrena el modelo y evalúa el rendimiento en los conjuntos de entrenamiento y test."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Training
    model.train()
    device = 'cuda'
    model = model.to(device)
    for epoch in range(num_epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        train_loss, correct_train = 0, 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct_train += pred.eq(target.view_as(pred)).sum().item()

        test_loss, correct_test = 0, 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct_test += pred.eq(target.view_as(pred)).sum().item()

    train_accuracy = correct_train / len(train_loader.dataset)
    test_accuracy = correct_test / len(test_loader.dataset)

    return train_loss / len(train_loader), train_accuracy, test_loss / len(test_loader), test_accuracy

def saveHeader(output_train_file, output_test_file):
    with open(output_train_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['units', 'epoch', 'train_loss', 'train_accuracy'])

    with open(output_test_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['units', 'test_loss', 'test_accuracy'])

def main():
    noise = 0.1 # % of label noise
    training_ds, test_ds = load_and_preprocess_mnist_data()
    classes, train_loader, test_loader = splitData(training_ds, test_ds, noise_level=noise, add_noise=True)
    output_train_file = '/mnt/homeGPU/jaruiz/mnist_train4k.txt'
    output_test_file = '/mnt/homeGPU/jaruiz/mnist_test4k.txt'
    saveHeader(output_train_file, output_test_file)

    # Entrenamiento y evaluación con diferentes números de unidades
    num_classes = len(classes)
    num_epochs=1000

    # Tiempo de inicio 
    inicio = time.time()

    num_units = np.arange(1, 101)
    for units in num_units:
        model = create_2nn(units, num_classes)
        
        train_loss, train_accuracy, test_loss, test_accuracy = train_and_evaluate_model(model, train_loader, test_loader, num_epochs)

        with open(output_train_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([units, num_epochs, train_loss, train_accuracy])

        with open(output_test_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([units, test_loss, test_accuracy])

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
