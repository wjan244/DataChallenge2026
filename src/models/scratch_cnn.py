import torch
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
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, padding_mode='circular')
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
    
    
# class ResBlock(torch.nn.Module):
# 	def __init__(self,inDim,hiddenDim,outDim,kernel):
# 		super().__init__()

# 		self.conv1 = torch.nn.Conv2d(inDim, hiddenDim, kernel,padding='same')
# 		self.conv2 = torch.nn.Conv2d(hiddenDim, outDim, kernel,padding='same')
# 		if inDim!=outDim:
# 			self.shortcut = torch.nn.Conv2d(inDim, outDim, 1,padding='same')
# 		else:
# 			self.shortcut = torch.nn.Identity()

# 	def forward(self, x):
# 		y = self.shortcut(x)
# 		x = F.relu(self.conv1(x))
# 		x = F.relu(self.conv2(x))
# 		return x + y

# # class ResNet(torch.nn.Module):
# # 	def __init__(self,p=0.5):
# # 		super().__init__()
# # 		self.ResBlock1 = ResBlock(inDim=1, hiddenDim=2, outDim=3, kernel=3)
# # 		self.pool = torch.nn.MaxPool2d(2, 2)
# # 		self.ResBlock2 = ResBlock(inDim=3, hiddenDim=4, outDim=6, kernel=3)
# # 		self.ResBlock3 = ResBlock(inDim=6, hiddenDim=9, outDim=12, kernel=3)
# # 		self.fc = torch.nn.Linear(6*6*12, 2)

# # 	def forward(self, x):
# # 		x = x.view(-1,1,48,48)
# # 		x = self.pool(self.ResBlock1(x))
# # 		x = self.pool(self.ResBlock2(x))
# # 		x = self.pool(self.ResBlock3(x))
# # 		x = torch.flatten(x, 1) # flatten all dimensions except batch
# # 		x = self.fc(x)
# # 		return x

# # class DropNet(torch.nn.Module):
# # 	def __init__(self,DropOutRate):
# # 		super().__init__()
# # 		self.conv1 = torch.nn.Conv2d(1, 3, 3, padding='same')
# # 		self.pool = torch.nn.MaxPool2d(2, 2)
# # 		self.conv2 = torch.nn.Conv2d(3, 6, 3, padding='same')
# # 		self.conv3 = torch.nn.Conv2d(6, 12, 3, padding='same')
# # 		self.fc = torch.nn.Linear(6 * 6 * 12, 2)
# # 		self.drop = torch.nn.Dropout(DropOutRate)

# # 	def forward(self, x):
# # 		x = x.view(-1, 1, 48, 48)
# # 		x = self.pool(F.relu(self.conv1(x)))
# # 		x = self.drop(x)
# # 		x = self.pool(F.relu(self.conv2(x)))
# # 		x = self.drop(x)
# # 		x = self.pool(F.relu(self.conv3(x)))
# # 		x = self.drop(x)
# # 		x = torch.flatten(x, 1)  # flatten all dimensions except batch
# # 		x = self.fc(x)
# # 		return x
