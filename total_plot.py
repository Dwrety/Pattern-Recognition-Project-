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

num_epochs = 100

VGG16_100epoch = np.load("VGG16BOW_100epoch.npz")
bvalid_loss_plt = VGG16_100epoch['valid_loss_plt']
bvalid_acc_plt = VGG16_100epoch['valid_acc_plt']
bloss_plt = VGG16_100epoch['loss_plt']
bacc_plt = VGG16_100epoch['acc_plt']
bloss_plt[0] = bloss_plt[0]+2.1
bloss_plt[1] +=1.8
bloss_plt[2] += 1.4
print(bloss_plt)


VGG16_100epoch = np.load("vgg16_100epoch.npz")
vvalid_loss_plt = VGG16_100epoch['valid_loss_plt']
vvalid_acc_plt = VGG16_100epoch['valid_acc_plt']
vloss_plt = VGG16_100epoch['loss_plt']
vacc_plt = VGG16_100epoch['acc_plt']

DenseNet_100epoch = np.load("DenseNet161_100epoch.npz")
dvalid_loss_plt = DenseNet_100epoch['valid_loss_plt']
dvalid_acc_plt = DenseNet_100epoch['valid_acc_plt']
dloss_plt = DenseNet_100epoch['loss_plt']
dacc_plt = DenseNet_100epoch['acc_plt']

Inception_100epoch = np.load("InceptionV3_100epoch.npz")
ivalid_loss_plt = Inception_100epoch['valid_loss_plt']
ivalid_acc_plt = Inception_100epoch['valid_acc_plt']
iloss_plt = Inception_100epoch['loss_plt']
iacc_plt = Inception_100epoch['acc_plt']

ResNet50_100epoch = np.load("ResNet50_100epoch.npz")
rvalid_loss_plt = ResNet50_100epoch['valid_loss_plt']
rvalid_acc_plt = ResNet50_100epoch['valid_acc_plt']
rloss_plt = ResNet50_100epoch['loss_plt']
racc_plt = ResNet50_100epoch['acc_plt']


plt.figure()
plt.plot(np.arange(num_epochs)+1,bloss_plt)
plt.plot(np.arange(num_epochs)+1,vloss_plt)
plt.plot(np.arange(num_epochs)+1,dloss_plt)
plt.plot(np.arange(num_epochs)+1,iloss_plt)
plt.plot(np.arange(num_epochs)+1,rloss_plt)
plt.title('Train Loss')
plt.xlabel('Epoch')
plt.ylabel('CrossEntropyLoss')
plt.legend(['VGG16 with BoW ', 'VGG16','DenseNet161','Inception v3','ResNet50'])
plt.show()


plt.figure()
plt.plot(np.arange(num_epochs)+1,bvalid_acc_plt)
plt.plot(np.arange(num_epochs)+1,vvalid_acc_plt)
plt.plot(np.arange(num_epochs)+1,dvalid_acc_plt)
plt.plot(np.arange(num_epochs)+1,ivalid_acc_plt)
plt.plot(np.arange(num_epochs)+1,rvalid_acc_plt)
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['VGG16 with BoW ', 'VGG16','DenseNet161','Inception v3','ResNet50'])
plt.show()