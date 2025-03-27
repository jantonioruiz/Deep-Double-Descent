import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class TwoLayerNN(nn.Module):
    def __init__(self, input_size, size=1, num_classes=10):
        super(TwoLayerNN, self).__init__()
        self.flatten = nn.Flatten()  
        self.fc1 = nn.Linear(input_size, size)  
        self.fc2 = nn.Linear(size, num_classes) 

    def forward(self, x):
        x = self.flatten(x)  
        x = torch.relu(self.fc1(x))  
        x = self.fc2(x)  
        return x

class DeepNN(nn.Module):
    def __init__(self, input_size, size=1, num_classes=10):
        super(DeepNN, self).__init__()

        hidden_size = (size + 1) // 2
        self.flatten = nn.Flatten()

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 2*hidden_size), nn.ReLU(),
            nn.Linear(2*hidden_size, 2*hidden_size), nn.ReLU(),
            nn.Linear(2*hidden_size, 2*hidden_size), nn.ReLU(),
            nn.Linear(2*hidden_size, 2*hidden_size), nn.ReLU(),
            nn.Linear(2*hidden_size, 4*hidden_size), nn.ReLU(),
            nn.Linear(4*hidden_size, num_classes) 
        )

    def forward(self, x):
        x = self.flatten(x)  
        x = self.layers(x)    
        return x

class ThreeLayerCNN(nn.Module):
    def __init__(self, in_channels, k, num_classes):
        super(ThreeLayerCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, k, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(k)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)  
        
        self.conv2 = nn.Conv2d(k, 2 * k, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(2 * k)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)  
        
        self.conv3 = nn.Conv2d(2 * k, 4 * k, kernel_size=5, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(4 * k)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(4)  
        
        self.flatten = nn.Flatten()  
        self.fc = nn.Linear(4 * k, num_classes)  
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        
        x = self.flatten(x) 
        x = self.fc(x)
        return x

class DeepCNN(nn.Module):
  def __init__(self, in_channels, k, num_classes):
    super(DeepCNN, self).__init__()
    self.conv1 = nn.Conv2d(in_channels, k, kernel_size=5, stride=1, padding=1)
    self.bn1 = nn.BatchNorm2d(k)
    self.relu1 = nn.ReLU()

    self.conv2 = nn.Conv2d(k, k, kernel_size=5, stride=1, padding=1)
    self.bn2 = nn.BatchNorm2d(k)
    self.relu2 = nn.ReLU()

    self.conv3 = nn.Conv2d(k, k, kernel_size=5, stride=1, padding=1)
    self.bn3 = nn.BatchNorm2d(k)
    self.relu3 = nn.ReLU()

    self.conv4 = nn.Conv2d(k, k, kernel_size=5, stride=1, padding=1)
    self.bn4 = nn.BatchNorm2d(k)
    self.relu4 = nn.ReLU()

    self.conv5 = nn.Conv2d(k, k, kernel_size=5, stride=1, padding=1)
    self.bn5 = nn.BatchNorm2d(k)
    self.relu5 = nn.ReLU()

    self.conv6 = nn.Conv2d(k, k, kernel_size=5, stride=1, padding=1)
    self.bn6 = nn.BatchNorm2d(k)
    self.relu6 = nn.ReLU()

    self.conv7 = nn.Conv2d(k, k, kernel_size=5, stride=1, padding=1)
    self.bn7 = nn.BatchNorm2d(k)
    self.relu7 = nn.ReLU()

    self.conv8 = nn.Conv2d(k, k, kernel_size=5, stride=1, padding=1)  
    self.bn8 = nn.BatchNorm2d(k)
    self.relu8 = nn.ReLU()

    self.conv9 = nn.Conv2d(k, k, kernel_size=5, stride=1, padding=1)
    self.bn9 = nn.BatchNorm2d(k)
    self.relu9 = nn.ReLU()

    self.conv10 = nn.Conv2d(k, k, kernel_size=5, stride=1, padding=1)
    self.bn10 = nn.BatchNorm2d(k)
    self.relu10 = nn.ReLU()

    self.conv11 = nn.Conv2d(k, k, kernel_size=3, stride=1)
    self.bn11 = nn.BatchNorm2d(k)
    self.relu11 = nn.ReLU()

    self.conv12 = nn.Conv2d(k, k, kernel_size=3, stride=1)
    self.bn12 = nn.BatchNorm2d(k)
    self.relu12 = nn.ReLU()

    self.conv13 = nn.Conv2d(k, k, kernel_size=3, stride=1)
    self.bn13 = nn.BatchNorm2d(k)
    self.relu13 = nn.ReLU()

    self.conv14 = nn.Conv2d(k, k, kernel_size=3, stride=1)
    self.bn14 = nn.BatchNorm2d(k)
    self.relu14 = nn.ReLU()

    self.pool = nn.MaxPool2d(4)  

    self.flatten = nn.Flatten()
    self.fc = nn.Linear(k, num_classes)

  def forward(self, x):
    x = self.relu1(self.bn1(self.conv1(x)))
    x = self.relu2(self.bn2(self.conv2(x)))
    x = self.relu3(self.bn3(self.conv3(x)))
    x = self.relu4(self.bn4(self.conv4(x)))
    x = self.relu5(self.bn5(self.conv5(x)))
    x = self.relu6(self.bn6(self.conv6(x)))
    x = self.relu7(self.bn7(self.conv7(x)))
    x = self.relu8(self.bn8(self.conv8(x)))
    x = self.relu9(self.bn9(self.conv9(x)))
    x = self.relu10(self.bn10(self.conv10(x)))
    x = self.relu11(self.bn11(self.conv11(x)))
    x = self.relu12(self.bn12(self.conv12(x)))
    x = self.relu13(self.bn13(self.conv13(x)))
    x = self.pool(self.relu14(self.bn14(self.conv14(x))))  

    x = self.flatten(x)
    x = self.fc(x)
    return x
  
class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, **kwargs):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channels, k=64, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = k
        c = k

        self.conv1 = nn.Conv2d(in_channels, c, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, c, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2*c, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4*c, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 8*c, num_blocks[3], stride=2)
        self.linear = nn.Linear(8*c*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        # eg: [2, 1, 1, ..., 1]. Only the first one downsamples.
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)
    
def createNet(net_type, **kwargs):
    param_defaults = {
        'TwoLayerNN': {'input_size': 28*28, 'size': 64, 'num_classes': 10},
        'DeepNN': {'input_size': 28*28, 'size': 64, 'num_classes': 10},
        'ThreeLayerCNN': {'in_channels': 1, 'k': 64, 'num_classes': 10},
        'DeepCNN': {'in_channels': 1, 'k': 64, 'num_classes': 10},
        'PreActResNet': {'block': PreActBlock, 'num_blocks': [2, 2, 2, 2], 'in_channels': 3, 'k': 64, 'num_classes': 10},
        'ResNet18': {'num_classes': 10}
    }
    
    if net_type not in param_defaults:
        raise ValueError(f"Red desconocida: {net_type}. Opciones disponibles: {list(param_defaults.keys())}")
    
    params = {**param_defaults[net_type], **kwargs}  

    return globals()[net_type](**params)