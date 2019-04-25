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
dataset = np.load("final_datapack.npz")

train_data = dataset['train_data']
y_train = dataset['y_train']
train_histograms = dataset['train_histograms']

test_data = dataset['test_data']
y_test = dataset['y_test']
test_histograms = dataset['test_histograms']

selected_breed_list = dataset['selected_breed_list']

batch_size = 36
num_epochs = 100

train_x = torch.from_numpy(np.asarray(train_data)).float()
train_y = torch.from_numpy(np.asarray(y_train)).long()
train_x_bow = torch.from_numpy(np.asarray(train_histograms)).float()
valid_x = torch.from_numpy(np.asarray(test_data)).float()
valid_y = torch.from_numpy(np.asarray(y_test)).long()
valid_x_bow = torch.from_numpy(np.asarray(test_histograms)).float()

image_datasets = {'train':train_x, 'validation':valid_x}
train_data_torch = torch.utils.data.TensorDataset(train_x,train_y,train_x_bow)
train_loader = torch.utils.data.DataLoader(dataset=train_data_torch, batch_size=batch_size, shuffle=True)
valid_data_torch = torch.utils.data.TensorDataset(valid_x,valid_y,valid_x_bow)
valid_loader = torch.utils.data.DataLoader(dataset=valid_data_torch, batch_size=batch_size, shuffle=True)
dataloaders = {'train':train_loader, 'validation':valid_loader}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation']}
num_batch = len(train_loader)
num_classes = 24


class Vgg16(nn.Module):
	def __init__(self,pretrained_model):
		super(Vgg16,self).__init__()
		self.features = pretrained_model.features
		for p in self.features.parameters():
			p.requires_grad = False
		self.classifier = nn.Sequential(*[pretrained_model.classifier[i] for i in range(5)])

		# self.modelName = 'VGG-16'

	def forward(self, x):
		f = self.features(x)
		f = f.view(f.size(0),-1)
		# nnn*4096
		y = self.classifier(f)
		return y


class Vgg16_BOW(nn.Module):
	def __init__(self, num_classes):
		super(Vgg16_BOW, self).__init__()
		self.fc = nn.Sequential(nn.Linear(4096+4200, num_classes))

	def forward(self, x, y):
		# print(x.size())
		# print(y.size())
		# y = 200*y
		f = torch.cat((x,y),1)
		z = self.fc(f)
		return z


vgg16 = Vgg16(Models.vgg16(pretrained=True)).to(device)
vgg16_bow = Vgg16_BOW(24).to(device)

criterion = nn.CrossEntropyLoss()
# optimizer_1 = torch.optim.Adam(vgg16.parameters(), lr=0.001)
# optimizer_2 = torch.optim.Adam(vgg16_bow.parameters(), lr=0.001)
optimizer_1 = torch.optim.SGD(vgg16.parameters(), lr=0.001, momentum=0.9)
optimizer_2 = torch.optim.SGD(vgg16_bow.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler_1 = lr_scheduler.StepLR(optimizer_1, step_size=20, gamma=0.9)
exp_lr_scheduler_2 = lr_scheduler.StepLR(optimizer_2, step_size=20, gamma=0.9)

# print(vgg16)

valid_loss_plt = []
valid_acc_plt = []
loss_plt = []
acc_plt = []
# training 
train = True
# print(vgg16_bow)


def train_model(vgg16, vgg16_bow, criterion, optimizer_1,optimizer_2, scheduler_1, scheduler_2, num_epochs=25):
	since = time.time()

	best_model_wts_1 = copy.deepcopy(vgg16.state_dict())
	best_model_wts_2 = copy.deepcopy(vgg16_bow.state_dict())
	best_acc = 0.0

	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch+1, num_epochs))
		print('-' * 30)

		# Each epoch has a training and validation phase
		for phase in ['train', 'validation']:
			if phase == 'train':
				scheduler_1.step()
				scheduler_2.step()
				vgg16.train()
				vgg16_bow.train()  # Set model to training mode
			else:
				vgg16.eval()
				vgg16_bow.eval()   # Set model to evaluate mode

			running_loss = 0.0
			running_corrects = 0

			# Iterate over data.
			for images, labels, bow_features in dataloaders[phase]:
				images = images.to(device)
				labels = labels.to(device)
				bow_features = bow_features.to(device)

				# zero the parameter gradients
				optimizer_1.zero_grad()
				optimizer_2.zero_grad()

				# forward
				# track history if only in train
				with torch.set_grad_enabled(phase == 'train'):
					x = vgg16(images)
					outputs = vgg16_bow(x, bow_features)
					# print(outputs.size())
					_, preds = torch.max(outputs, 1)
					loss = criterion(outputs, labels)

					# backward + optimize only if in training phase
					if phase == 'train':
						loss.backward()
						optimizer_1.step()
						optimizer_2.step()

				# statistics
				running_loss += loss.item() * images.size(0)
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
				best_model_wts_1 = copy.deepcopy(vgg16.state_dict())
				best_model_wts_2 = copy.deepcopy(vgg16_bow.state_dict())

		print()

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))
	print('Best val Acc: {:4f}'.format(best_acc))

	# load best model weights
	vgg16.load_state_dict(best_model_wts_1)
	vgg16_bow.load_state_dict(best_model_wts_2)
	return vgg16, vgg16_bow



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



if False:
	vgg16, vgg16_bow = train_model(vgg16,vgg16_bow, criterion, optimizer_1, optimizer_2, exp_lr_scheduler_1,
									exp_lr_scheduler_2, num_epochs=num_epochs)
	torch.save(vgg16.state_dict(), 'VGG16bow1_100epoch.ckpt')
	torch.save(vgg16_bow.state_dict(),'VGG16bow2_100epoch.ckpt')

	valid_loss_plt = np.asarray(valid_loss_plt)
	valid_acc_plt = np.asarray(valid_acc_plt)
	loss_plt = np.asarray(loss_plt)
	acc_plt = np.asarray(acc_plt)
	np.savez("VGG16BOW_100epoch.npz",valid_loss_plt=valid_loss_plt,valid_acc_plt=valid_acc_plt,loss_plt=loss_plt,acc_plt=acc_plt)

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


if True:
	selected_breed_list_correct = ['scottish_deerhound', 'maltese_dog', 'afghan_hound' ,'entlebucher',
									 'bernese_mountain_dog', 'shih-tzu' ,'great_pyrenees', 'pomeranian', 'basenji',
									 'samoyed' ,'airedale', 'tibetan_terrier' ,'leonberg' ,'cairn', 'beagle',
									 'japanese_spaniel', 'australian_terrier' ,'blenheim_spaniel',
									 'miniature_pinscher' ,'irish_wolfhound' ,'lakeland_terrier' ,'saluki',
									 'papillon' ,'whippet']
	correct_dict = {}								 
	for i in range(24):
		correct_dict[selected_breed_list_correct[i]] = i
	# print(correct_dict)	

	# print(len(selected_breed_list_correct))
	VGG16_100epoch = np.load("VGG16BOW_100epoch.npz")
	print(VGG16_100epoch.files)
	valid_loss_plt = VGG16_100epoch['valid_loss_plt']
	valid_acc_plt = VGG16_100epoch['valid_acc_plt']
	loss_plt = VGG16_100epoch['loss_plt']
	acc_plt = VGG16_100epoch['acc_plt']

	plt.figure()
	plt.plot(np.arange(num_epochs)+1,loss_plt)
	plt.plot(np.arange(num_epochs)+1,valid_loss_plt)
	plt.title('VGG16 with BOW Loss')
	plt.xlabel('Epoch')
	plt.ylabel('CrossEntropyLoss')
	plt.legend(['Train Loss', 'Validation Loss'])
	plt.show()

	plt.figure()
	plt.plot(np.arange(num_epochs)+1,acc_plt)
	plt.plot(np.arange(num_epochs)+1,valid_acc_plt)
	plt.title('VGG16 with BOW Accuracy')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.legend(['Train Accuracy','Validation Accuracy'])
	plt.show()


	vgg16.load_state_dict(torch.load('VGG16bow1_100epoch.ckpt'))
	vgg16_bow.load_state_dict(torch.load('VGG16bow2_100epoch.ckpt'))
	vgg16.eval()
	vgg16_bow.eval()
	# print()
	test_label = []
	predicted_label = []

	for inputs, labels, bow_features in valid_loader:
		# print(labels.data.numpy())
		yyy = labels.data.numpy()
		for yy in yyy:
			# print(correct_dict[selected_breed_list[yy]])
			test_label.append(correct_dict[selected_breed_list[yy]])

		# test_label = np.concatenate((test_label,labels.data.numpy()))
		inputs = inputs.to(device)
		labels = labels.to(device)
		bow_features = bow_features.to(device)

		x = vgg16(inputs)
		outputs = vgg16_bow(x,bow_features)
		_, preds = torch.max(outputs, 1)
		yyy = preds.cpu().data.numpy()
		for yy in yyy:
			predicted_label.append(correct_dict[selected_breed_list[yy]])
		# print(preds.items())
	# print(test_label.shape)
	# print(predicted_label.shape)
	cnf_matrix = confusion_matrix(test_label, predicted_label, labels = np.arange(24))
	np.set_printoptions(precision=2)
	plt.figure(figsize=(8,8))
	plot_confusion_matrix(cnf_matrix, classes=selected_breed_list_correct, normalize=True,
							title='VGG16 with BOW confusion matrix')

	plt.show()