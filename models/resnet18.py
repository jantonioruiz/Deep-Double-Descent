import csv
import time
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

# ResNet18 oficial implementation
"""
    The following ResNet implementations are the official version of Pytorch library
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# MODIFIED! We use [k, 2k, 4k, 8k] filters for each conv layer
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, k=64):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        # MODIFIED! Start with k channels instead of 64
        self.inplanes = k
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        # MODIFIED! Modify BatchNorm to use k
        self.conv1 = nn.Conv2d(3, k, kernel_size=7, stride=2, padding=3,
                               bias=False)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # MODIFIED! Modify all layers to use [k, 2k, 4k, 8k]
        self.layer1 = self._make_layer(block, k, layers[0])
        self.layer2 = self._make_layer(block, 2*k, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 4*k, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 8*k, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # MODIFIED! Modify the FC (8k)
        self.fc = nn.Linear(8*k * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

# Function to create the model (k=64 is equivalent to ResNet18 architecture)
def resnet18(k=64, num_classes=1000, **kwargs):
    """Constructs a ResNet-18 model."""

    model = ResNet(BasicBlock, [2, 2, 2, 2], k=k, num_classes=num_classes, **kwargs)
    return model

# Function to train the model
def train_model(model, train_loader, criterion, optimizer, output_file, num_epochs=10, model_dimension=None):
    # Model to training mode
    model.train()  

    # We are going to use gpu ('cuda') only
    device = 'cuda'

    # We move the model to the gpu
    model = model.to(device)

    for epoch in range(num_epochs):
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

        epoch_loss = round(running_loss / len(train_loader), 4)
        epoch_accuracy = round(correct / total, 4)

    print(f"Model dimension: {model_dimension:d}, Epoch [{epoch + 1}/{num_epochs}], Train loss: {epoch_loss:.4f}, Train accuracy: {epoch_accuracy:.4f}")

    # save results
    with open(output_file, mode='a', newline='') as file:
      writer = csv.writer(file)
      writer.writerow([model_dimension, epoch + 1, epoch_loss, epoch_accuracy])

# Function to evaluate the model
def test_model(model, test_loader, criterion, output_file, model_dimension=None):
    # Model to evaluation mode
    model.eval()

    # We are going to use gpu ('cuda') only
    device = 'cuda'

    # We move the model to the gpu
    model = model.to(device)

    correct = 0
    total = 0
    running_loss = 0.0

    # Disabling gradient calculation for efficiency, as we don’t need to update parameters
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = round(correct / total, 4)
    test_loss = round(running_loss / len(test_loader),4)
    print(f"Model dimension: {model_dimension:d}, Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}")

    # save results
    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([model_dimension, test_loss, test_accuracy])  

# Function to get the CIFAR10 dataset with some transformations (in order to augment the data)
def getData():
    train_transforms = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()])

    test_transforms = transforms.Compose([transforms.ToTensor()])

    training_ds = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transforms)
    test_ds = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transforms)

    return training_ds, test_ds

# Function to add label noise (noise_level is equivalent to the % of desired noise)
def add_label_noise(dataset, noise_level=0.2):
    num_samples = len(dataset)
    num_noisy = int(noise_level * num_samples)
    
    indices = np.random.choice(num_samples, num_noisy, replace=False)
    
    for i in indices:
        original_label = dataset.targets[i]
        
        noisy_label = original_label
        while noisy_label == original_label:
            noisy_label = np.random.randint(0, len(dataset.classes))
        
        dataset.targets[i] = noisy_label

# Function to split the data in dataloaders (train and test)
def splitData(training_ds, test_ds, noise_level=0.2, add_noise=False):
    torch.manual_seed(42)

    if add_noise:
        add_label_noise(training_ds, noise_level=noise_level)

    classes = training_ds.classes
    batch_size = 128

    train_loader = torch.utils.data.DataLoader(training_ds, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return classes, train_loader, test_loader

# Function to save the header of the outputs files
def saveHeader(output_train_file, output_test_file):
    with open(output_train_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['model_dimension', 'epoch', 'train_loss', 'train_accuracy'])

    with open(output_test_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['model_dimension', 'test_loss', 'test_accuracy'])

# main function
def main():
    training_ds, test_ds = getData()
    classes, train_loader, test_loader = splitData(training_ds, test_ds, noise_level=0.2, add_noise=True)
    output_train_file = '/mnt/homeGPU/jaruiz/resnet18_train_noise.txt'
    output_test_file = '/mnt/homeGPU/jaruiz/resnet18_test_noise.txt'
    saveHeader(output_train_file, output_test_file)

    num_classes = len(classes)
    criterion = nn.CrossEntropyLoss() 

    # init time
    inicio = time.time()

    # Training and testing
    for k in range(1, 65):
        model = resnet18(k=k, num_classes=num_classes)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        train_model(model, train_loader, criterion=criterion, optimizer=optimizer, output_file=output_train_file, num_epochs=25, model_dimension=k)
        test_model(model, test_loader, criterion=criterion, output_file=output_test_file, model_dimension=k)
        print()

    # end time
    fin = time.time()

    # total time
    tiempo_total = int(fin - inicio)
    horas = tiempo_total // 3600
    minutos = (tiempo_total % 3600) // 60
    segundos = tiempo_total % 60

    # save time result
    with open(output_train_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([])
        writer.writerow([f"La ejecución ha durado {horas} hora(s), {minutos} minuto(s) y {segundos} segundo(s)."])

if __name__ == "__main__":
    main()