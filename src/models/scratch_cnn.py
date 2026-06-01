import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#todo
# dropout
# init weights
# cos annealing
# resnet
# load weight

def _init_weights(self):
    rng_state = torch.get_rng_state()
    torch.seed()
    for m in self.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            torch.nn.init.zeros_(m.bias)
    torch.set_rng_state(rng_state)

class _ConvBlock(torch.nn.Module):
    def __init__(self,in_channels, out_channels, dropout):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, padding_mode='reflect')
        self.bn = torch.nn.BatchNorm2d(out_channels) # cause issues on MPs with compile?
        self.activation = F.relu        
        self.drop = torch.nn.Dropout2d(p=dropout)
        # self.pool = torch.nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.drop(x)
        #x = self.pool(x)
        
        return x

    
class ConvNet(torch.nn.Module):
    def __init__(self, num_classes = 1, dropout=0.2):
        super().__init__()
    
        self.conv1 = _ConvBlock(3, 32, dropout) #224-112
        self.conv2 = _ConvBlock(32, 64, dropout) #112->56
        self.conv3 = _ConvBlock(64, 128, dropout) #56->28
        self.conv4 = _ConvBlock(128, 256, dropout) #28-> 14
        self.pool = torch.nn.AdaptiveAvgPool2d(1) #14x14 -> 1
        self.drop_fc = torch.nn.Dropout(p=dropout)
        self.fc = torch.nn.Linear(256, num_classes)
        _init_weights(self)

    def forward(self, x):
        # x = x.view(-1,3,224,224)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = x.flatten(1) # B 256 1 1 -> B 256
        x = self.drop_fc(x)
        x = self.fc(x)       

        return x
    
    
class ResBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=stride, bias=False, padding_mode='reflect')
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, 3, padding=1, stride=1, bias=False, padding_mode='reflect')
        self.bn2 = torch.nn.BatchNorm2d(out_channels)

        if in_channels!=out_channels:
            self.shortcut = torch.nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding='same', padding_mode='reflect')
        else:
            self.shortcut = torch.nn.Identity()

    def forward(self, x):
        y = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + y)

def _make_layer(in_channels, out_channels, num_blocks, stride):
    layers =[ResBlock(in_channels, out_channels, stride=stride)]
    for _ in range(1, num_blocks):
        layers.append(ResBlock(out_channels, out_channels, stride=1))
    
    return torch.nn.Sequential(*layers)


class ResNet18(torch.nn.Module):
    def __init__(self,p=0.3, num_classes=1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1,padding_mode='circular')
        ) # 64x56x56
        
        self.layer1 = _make_layer(64, 64, num_blocks=2, stride=1) # 64x56x56
        self.layer2 = _make_layer(64, 128, num_blocks=2, stride=2) # 128x28x28
        self.layer3 = _make_layer(128, 256, num_blocks=2, stride=2) # 256x14x14
        # self.layer4 = _make_layer(256, 512, num_blocks=2, stride=2) # 512x7x7
        
        self.pool = nn.AdaptiveAvgPool2d((1,1)) #256x1x1

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512,256),
            nn.RelU(),
            nn.Dropout(p),
            nn.Linear(256, num_classes) 
        ) # no sigmoid done in occlusion wrapper
        _init_weights(self)
        
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = self.head(x)
        return x
