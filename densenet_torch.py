import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torchvision
import torch
import torchvision.models as Models
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
import time
import copy
import glob
import itertools
from sklearn.metrics import confusion_matrix


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dataset = np.load("train_dataset.npz")

X_train = dataset['X_train']
X_valid = dataset['X_valid']
Y_train = dataset['Y_train']
Y_valid = dataset['Y_valid']
selected_breed_list = dataset['selected_breed_list']

batch_size = 36
num_epochs = 100


train_x = torch.from_numpy(np.asarray(X_train)).float()
train_y = torch.from_numpy(np.where(Y_train == 1)[1])
valid_x = torch.from_numpy(np.asarray(X_valid)).float()
valid_y = torch.from_numpy(np.where(Y_valid == 1)[1])
image_datasets = {'train':train_x, 'validation':valid_x}
train_data_torch = torch.utils.data.TensorDataset(train_x,train_y)
train_loader = torch.utils.data.DataLoader(dataset=train_data_torch, batch_size=batch_size, shuffle=True)
valid_data_torch = torch.utils.data.TensorDataset(valid_x,valid_y)
valid_loader = torch.utils.data.DataLoader(dataset=valid_data_torch, batch_size=batch_size, shuffle=True)
dataloaders = {'train':train_loader, 'validation':valid_loader}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation']}
num_batch = len(train_loader)
num_classes = Y_train.shape[1]


# class ResNet50(nn.Module):
# 	def __init__(self,pretrained_model,num_classes):
# 		super(ResNet50,self).__init__()
# 		self.conv1 = pretrained_model.conv1
# 		self.inplanes = 64
# 		self.bn1 = pretrained_model.bn1
# 		self.maxpool = pretrained_model.maxpool
# 		self.layer1 = pretrained_model.layer1
# 		self.layer2 = pretrained_model.layer2
# 		self.layer3 = pretrained_model.layer3
# 		self.layer4 = pretrained_model.layer4
# 		self.avgpool = pretrained_model.avgpool
# 		self.fc = nn.Linear(512*block.expansion, num_classes)

# 		# self.modelName = 'VGG-16'

# 	def forward(self, x):

# 		f = self.features(x)

# 		f = f.view(f.size(0),-1)

# 		y = self.classifier(f)
# 		return y 	

DenseNet = Models.densenet161(pretrained=True)
for param in DenseNet.parameters():
	param.requires_grad = False
num_features = DenseNet.classifier.in_features
DenseNet.classifier = nn.Linear(num_features, num_classes)
DenseNet.classifier = nn.Sequential(nn.Linear(num_features, num_classes),
							nn.Softmax(dim=1))

model = DenseNet.to(device)
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)


valid_loss_plt = []
valid_acc_plt = []
loss_plt = []
acc_plt = []


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
	since = time.time()

	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0

	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch+1, num_epochs))
		print('-' * 30)

		# Each epoch has a training and validation phase
		for phase in ['train', 'validation']:
			if phase == 'train':
				scheduler.step()
				model.train()  # Set model to training mode
			else:
				model.eval()   # Set model to evaluate mode

			running_loss = 0.0
			running_corrects = 0

			# Iterate over data.
			for inputs, labels in dataloaders[phase]:
				inputs = inputs.to(device)
				labels = labels.to(device)

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward
				# track history if only in train
				with torch.set_grad_enabled(phase == 'train'):
					outputs = model(inputs)
					# print(outputs)
					_, preds = torch.max(outputs, 1)
					loss = criterion(outputs, labels)

					# backward + optimize only if in training phase
					if phase == 'train':
						loss.backward()
						optimizer.step()

				# statistics
				running_loss += loss.item() * inputs.size(0)
				running_corrects += torch.sum(preds == labels.data)

			epoch_loss = running_loss / dataset_sizes[phase]
			epoch_acc = running_corrects.double() / dataset_sizes[phase]

			print('{} Loss: {:.4f} Acc: {:.4f}'.format(
				phase, epoch_loss, epoch_acc))
			if phase == 'train':
				loss_plt.append(epoch_loss)
				acc_plt.append(epoch_acc)

			# deep copy the model
			if phase == 'validation':
				valid_loss_plt.append(epoch_loss)
				valid_acc_plt.append(epoch_acc)

			if phase == 'validation' and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = copy.deepcopy(model.state_dict())

		print()

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))
	print('Best val Acc: {:4f}'.format(best_acc))

	# load best model weights
	model.load_state_dict(best_model_wts)
	return model

if False:
	model = train_model(model, criterion, optimizer, exp_lr_scheduler,
						num_epochs=num_epochs)

	torch.save(model.state_dict(), 'DenseNet161_100epoch.ckpt')

	valid_loss_plt = np.asarray(valid_loss_plt)
	valid_acc_plt = np.asarray(valid_acc_plt)
	loss_plt = np.asarray(loss_plt)
	acc_plt = np.asarray(acc_plt)
	np.savez("DenseNet161_100epoch.npz",valid_loss_plt=valid_loss_plt,valid_acc_plt=valid_acc_plt,loss_plt=loss_plt,acc_plt=acc_plt)

	plt.figure()
	plt.plot(np.arange(num_epochs)+1,loss_plt)
	plt.plot(np.arange(num_epochs)+1,valid_loss_plt)
	plt.title('Loss')
	plt.legend(['Train Loss', 'Validation Loss'])
	plt.show()

	plt.figure()
	plt.plot(np.arange(num_epochs)+1,acc_plt)
	plt.plot(np.arange(num_epochs)+1,valid_acc_plt)
	plt.title('Accuracy')
	plt.legend(['Train Accuracy','Validation Accuracy'])
	plt.show()


def plot_confusion_matrix(cm, classes,
							normalize=False,
							title='Confusion matrix',
							cmap=plt.cm.Blues):
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	# print(classes)
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=90)
	plt.yticks(tick_marks, classes)

	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')


if True:

	DenseNet_100epoch = np.load("DenseNet161_100epoch.npz")

	valid_loss_plt = DenseNet_100epoch['valid_loss_plt']
	valid_acc_plt = DenseNet_100epoch['valid_acc_plt']
	loss_plt = DenseNet_100epoch['loss_plt']
	acc_plt = DenseNet_100epoch['acc_plt']

	plt.figure()
	plt.plot(np.arange(num_epochs)+1,loss_plt)
	plt.plot(np.arange(num_epochs)+1,valid_loss_plt)
	plt.title('DenseNet161 Loss')
	plt.xlabel('Epoch')
	plt.ylabel('CrossEntropyLoss')
	plt.legend(['Train Loss', 'Validation Loss'])
	plt.show()

	plt.figure()
	plt.plot(np.arange(num_epochs)+1,acc_plt)
	plt.plot(np.arange(num_epochs)+1,valid_acc_plt)
	plt.title('DenseNet161 Accuracy')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.legend(['Train Accuracy','Validation Accuracy'])
	plt.show()


	model.load_state_dict(torch.load('DenseNet161_100epoch.ckpt'))
	model.eval()
	# print()
	test_label = []
	predicted_label = []

	for inputs, labels in valid_loader:
		# print(labels.data.numpy())
		test_label = np.concatenate((test_label,labels.data.numpy()))
		inputs = inputs.to(device)
		labels = labels.to(device)

		outputs = model(inputs)
		_, preds = torch.max(outputs, 1)
		predicted_label= np.concatenate((predicted_label,preds.cpu().data.numpy()))
		# print(preds.items())
	# print(test_label.shape)
	# print(predicted_label.shape)
	cnf_matrix = confusion_matrix(test_label, predicted_label, labels = np.arange(24))
	np.set_printoptions(precision=2)
	plt.figure(figsize=(8, 8))
	plot_confusion_matrix(cnf_matrix, classes=selected_breed_list, normalize=True,
							title='DenseNet161 confusion matrix')

	plt.show()

