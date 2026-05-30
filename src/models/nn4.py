import torch
import DataSets as ds
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

LoadModel = False


class ConvNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 3, 3,padding='same')
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(3, 6, 3,padding='same')
        self.conv3 = torch.nn.Conv2d(6, 12, 3,padding='same')
        self.fc = torch.nn.Linear(6*6*12, 2)

    def forward(self, x, writer=None, iter=0):
        x = x.view(-1,1,48,48)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc(x)

        return x


def train_one_iter(model, optimizer, image, label, writer, iter):
	optimizer.zero_grad()
	y = model(image,writer,iter)
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
	model = torch.load('convnet%d.model'%n)
else:
	# simple_v2 = SimpleNet(train.dim)
	model = ConvNet()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.5e-1)

def load_image(name):
	f = open(name, 'rb')
	ima = np.empty([1, 2304], dtype=np.float32)
	ima[0,:] = np.fromfile(f, dtype=np.ubyte, count=2304)
	ima = (ima - 128.0) / 256.0
	f.close()
	return torch.from_numpy(ima)



nbIt = 3001
for iter in range(nbIt):
	ima, lab = train.NextTrainingBatch()
	loss = train_one_iter(model, optimizer, ima, lab, summary_writer if iter % 10 == 0 else None, iter)

	if iter % 100 == 0:
		print("iter= %6d - loss= %f" % (iter, loss))

	if iter % 500 == 0:
		Acc_Train_value = train.mean_accuracy(model)
		Acc_Test_value = test.mean_accuracy(model)
		print("iter= %6d - mean accuracy - train = %f  test = %f" % (iter, Acc_Train_value, Acc_Test_value))
		summary_writer.add_scalar("Acc_Train", Acc_Train_value, iter)
		summary_writer.add_scalar("Acc_Test", Acc_Test_value, iter)

	if iter % 2000 == 0:
		torch.save(model, 'convnet%d_it%d.model' % (n, iter))

torch.save(model, 'convnet%d.model'%n)
summary_writer.close()

exit()

ima = load_image("../Deep_Chuck/chuck48.raw")
label = model(ima)
lab = torch.softmax(label,-1).detach().numpy()
print("label = ",label.detach().numpy(),"      softmax label =",lab)
if lab[0,0]>.5:
	print("c'est un homme")
else:
	print("c'est une femme")