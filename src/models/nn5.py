

import torch
import DataSets as ds
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

LoadModel = False

n = 1
net ='resnet' # 'dropnet'  ou 'resnet'
experiment_name = '%s learned on %dk images'%(net,n)
train = ds.DataSet('../DataBases/data_%dk.bin'%n,'../DataBases/gender_%dk.bin'%n,n*1000)
test = ds.DataSet('../DataBases/data_test10k.bin','../DataBases/gender_test10k.bin',10000)

class ResBlock(torch.nn.Module):
	def __init__(self,inDim,hiddenDim,outDim,kernel):
		super().__init__()
		self.conv1 = torch.nn.Conv2d(inDim, hiddenDim, kernel,padding='same')
		self.conv2 = torch.nn.Conv2d(hiddenDim, outDim, kernel,padding='same')
		if inDim!=outDim:
			self.shortcut = torch.nn.Conv2d(inDim, outDim, 1,padding='same')
		else:
			self.shortcut = torch.nn.Identity()

	def forward(self, x):
		y = self.shortcut(x)
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		return x + y

class ResNet(torch.nn.Module):
	def __init__(self,p=0.5):
		super().__init__()
		self.ResBlock1 = ResBlock(inDim=1, hiddenDim=2, outDim=3, kernel=3)
		self.pool = torch.nn.MaxPool2d(2, 2)
		self.ResBlock2 = ResBlock(inDim=3, hiddenDim=4, outDim=6, kernel=3)
		self.ResBlock3 = ResBlock(inDim=6, hiddenDim=9, outDim=12, kernel=3)
		self.fc = torch.nn.Linear(6*6*12, 2)

	def forward(self, x):
		x = x.view(-1,1,48,48)
		x = self.pool(self.ResBlock1(x))
		x = self.pool(self.ResBlock2(x))
		x = self.pool(self.ResBlock3(x))
		x = torch.flatten(x, 1) # flatten all dimensions except batch
		x = self.fc(x)
		return x

class DropNet(torch.nn.Module):
	def __init__(self,DropOutRate):
		super().__init__()
		self.conv1 = torch.nn.Conv2d(1, 3, 3, padding='same')
		self.pool = torch.nn.MaxPool2d(2, 2)
		self.conv2 = torch.nn.Conv2d(3, 6, 3, padding='same')
		self.conv3 = torch.nn.Conv2d(6, 12, 3, padding='same')
		self.fc = torch.nn.Linear(6 * 6 * 12, 2)
		self.drop = torch.nn.Dropout(DropOutRate)

	def forward(self, x):
		x = x.view(-1, 1, 48, 48)
		x = self.pool(F.relu(self.conv1(x)))
		x = self.drop(x)
		x = self.pool(F.relu(self.conv2(x)))
		x = self.drop(x)
		x = self.pool(F.relu(self.conv3(x)))
		x = self.drop(x)
		x = torch.flatten(x, 1)  # flatten all dimensions except batch
		x = self.fc(x)
		return x

def train_one_iter(model, optimizer, image, label, writer, iter):
	model.train()
	optimizer.zero_grad()
	y = model(image)
	loss = F.cross_entropy(y,label)
	if writer is not None:
		writer.add_scalar('CrossEntropy', loss,iter)
	loss.backward()
	optimizer.step()
	return loss


print ("-----------------------------------------------------")
print ("-----------",experiment_name)
print ("-----------------------------------------------------")

summary_writer = SummaryWriter('summaries/%s' % experiment_name)
if LoadModel:
	model = torch.load(net)
else:
	if net == 'resnet':
		model = ResNet()
	else:
		model = DropNet(0.2)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.5e-1)

nbIt = 5001
for iter in range(nbIt):
	ima, lab = train.NextTrainingBatch()
	loss = train_one_iter(model, optimizer, ima, lab, summary_writer if iter % 10 == 0 else None, iter)

	if iter % 100 == 0:
		print("iter= %6d - loss= %f" % (iter, loss))

	if iter % 1000 == 0:
		Acc_Train_value = train.mean_accuracy(model)
		Acc_Test_value = test.mean_accuracy(model)
		print("iter= %6d - mean accuracy - train = %f  test = %f" % (iter, Acc_Train_value, Acc_Test_value))
		summary_writer.add_scalar("Acc_Train", Acc_Train_value, iter)
		summary_writer.add_scalar("Acc_Test", Acc_Test_value, iter)

	if iter % 10000 == 0:
		torch.save(model, net)

summary_writer.close()
