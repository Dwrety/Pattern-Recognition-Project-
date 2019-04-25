import numpy as np
import pandas as pd
import os
import cv2
import sys
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt 
import glob
import torch
# import torch.utils.data
import torchvision


# train data, validation data, test data generation program
# raw data folder
# df_train = pd.read_csv('../train/labels.csv')
IMG_SIZE = 224 
NUM_CLASSES = 24
# DATA_DIR = '../train'
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# # due to out computational limit, we chose only top 24 popular breeds
# # select top 16 popular dog breeds 
# selected_breed_list = list(df_train.groupby('breed').count().sort_values(by='id', ascending=False).head(NUM_CLASSES).index)
# print(selected_breed_list)

# for c in selected_breed_list:
# 	# print(c)
# 	path1 = os.path.join('C:/Pattern Recognition Dog Breed/Code/train/',c)
# 	path2 = os.path.join('C:/Pattern Recognition Dog Breed/Code/test/',c)
# 	# print(path)
# 	os.makedirs(path1)
# 	os.makedirs(path2)

# img_data_24 = df_train[df_train['breed'].isin(selected_breed_list)]
# target_series = pd.Series(img_data_24['breed'])
# one_hot = pd.get_dummies(target_series, sparse=True)
# label_24 = np.asarray(one_hot)
# print(img_data_24)




# y_train = label_24

# img_data_24_train, img_data_24_test = train_test_split(img_data_24,test_size=0.2, random_state=1)
# target_series_train = pd.Series(img_data_24_train['breed'])
# target_series_test = pd.Series(img_data_24_test['breed'])
# one_hot_train = pd.get_dummies(target_series_train, sparse=True)
# one_hot_test = pd.get_dummies(target_series_test, sparse=True)
# label_24_train = np.asarray(one_hot_train)
# label_24_test = np.asarray(one_hot_test)


# num_train = label_24_train.shape[0]
# num_test = label_24_test.shape[0]

# x_train = np.zeros((num_train,3,IMG_SIZE,IMG_SIZE))
# x_test = np.zeros((num_test,3,IMG_SIZE,IMG_SIZE))
# y_train = label_24_train
# y_test = label_24_test
# # print(y_train)

# i = 0 
# for img_id, breed in tqdm(img_data_24_train.values):
# 	img = cv2.imread('../train/{}.jpg'.format(img_id))
# 	path = os.path.join('C:/Pattern Recognition Dog Breed/Code/train/', breed+'/',img_id+'.jpg')
# 	cv2.imwrite(path, img)
# 	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 	img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))/255
# 	for j in range(3):
# 		img[:,:,j] = (img[:,:,j] - MEAN[j]) / STD[j]
# 	img = np.asarray([img[:,:,j] for j in range(3)]).astype(np.float32)
# 	x_train[i] = img 
# 	i += 1

# i = 0 
# for img_id, breed in tqdm(img_data_24_test.values):
# 	img = cv2.imread('../train/{}.jpg'.format(img_id))
# 	path = os.path.join('C:/Pattern Recognition Dog Breed/Code/test/', breed+'/',img_id+'.jpg')
# 	cv2.imwrite(path, img)
# 	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 	img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))/255
# 	for j in range(3):
# 		img[:,:,j] = (img[:,:,j] - MEAN[j]) / STD[j]
# 	img = np.asarray([img[:,:,j] for j in range(3)]).astype(np.float32)
# 	x_train[i] = img 
# 	i += 1

# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

# np.savez("train_pack_bow.npz",selected_breed_list=selected_breed_list,x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test)


# i = 0 
# for img_id, breed in tqdm(img_data_24_train.values):
# 	img = cv2.imread('../train/{}.jpg'.format(img_id))
# 	# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 	path = os.path.join('C:/Pattern Recognition Dog Breed/Code/test/', breed+'/',img_id+'.jpg')
# 	# path = os.path.join(path,'/')
# 	cv2.imwrite(path, img)	

# 	# print(img_id, breed)
# 	# plt.imshow(img)
	# plt.show()

	# img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))/255
	# for j in range(3):
	# 	img[:,:,j] = (img[:,:,j] - MEAN[j]) / STD[j]
	# img = np.asarray([img[:,:,j] for j in range(3)]).astype(np.float32)
	# x_train_raw[i] = img 
	# i += 1

# print(y_train_raw.shape)
# print(x_train_raw.shape)

# X_train, X_valid, Y_train, Y_valid = train_test_split(x_train_raw, y_train_raw, test_size=0.2, random_state=1)
# print(X_train.shape)
# print(X_valid.shape)

# np.savez("train_pack_299.npz",X_train=X_train,X_valid=X_valid,Y_train=Y_train,Y_valid=Y_valid,selected_breed_list=selected_breed_list)



class_names = [name[6:] for name in glob.glob('train/*')]
class_names = dict(zip(range(0,len(class_names)), class_names))

NUM_CLASSES = 24
selected_breed_list = []
for i in range(24):
	selected_breed_list.append(class_names[i])
print(selected_breed_list)



def load_dataset(path, num_per_class=-1):
	# train_labels, train_histograms, dictionary = trained_system['labels'], trained_system['features'], trained_system['dictionary']
	data = []
	labels = []
	for idx, class_name in class_names.items():
		img_path_class = glob.glob(path + class_name + '/*.jpg')
		# print(img_path_class)
		if num_per_class > 0:
			img_path_class = img_path_class[:num_per_class]
		labels.extend([idx]*len(img_path_class))	
		for filename in img_path_class:
			img = cv2.imread(filename)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))/255
			for j in range(3):
				img[:,:,j] = (img[:,:,j] - MEAN[j]) / STD[j]
			img = np.asarray([img[:,:,j] for j in range(3)]).astype(np.float32)
			# x_train[i] = img 
			data.append(img)
	labels = np.array(labels)		
	data = np.array(data)
	return data, labels

# load training dataset
train_data, y_train = load_dataset('train/', 140)
test_data, y_test = load_dataset('test/', 100)
# x_train = np.zeros((num_train,3,IMG_SIZE,IMG_SIZE))

batch_size = 36

# print(train_data.shape)
trained_system = np.load("trained_system.npz")
train_labels, train_histograms, dictionary = trained_system['labels'], trained_system['features'], trained_system['dictionary']
# print(train_labels == y_train)
test_histograms = np.load("test_system.npy")
# print(train_histograms)
# print(test_histograms)

np.savez("final_datapack.npz", train_data=train_data, y_train=y_train,train_histograms=train_histograms,
								test_data=test_data, y_test=y_test, test_histograms=test_histograms,selected_breed_list=selected_breed_list)

train_x = torch.from_numpy(np.asarray(train_data)).float()
train_y = torch.from_numpy(np.asarray(y_train))
train_x_bow = torch.from_numpy(np.asarray(train_histograms)).float()

valid_x = torch.from_numpy(np.asarray(test_data)).float()
valid_y = torch.from_numpy(np.asarray(y_test))
valid_x_bow = torch.from_numpy(np.asarray(test_histograms)).float()

train_data_torch = torch.utils.data.TensorDataset(train_x,train_y,train_x_bow)
train_loader = torch.utils.data.DataLoader(dataset=train_data_torch, batch_size=batch_size, shuffle=True)
valid_data_torch = torch.utils.data.TensorDataset(valid_x,valid_y,valid_x_bow)
valid_loader = torch.utils.data.DataLoader(dataset=valid_data_torch, batch_size=batch_size, shuffle=True)

# for images, labels, bow_features in train_loader:
# 	print(images.size())
# 	print(labels.size())
# 	print(bow_features.size())






# train_num = len(train_label)
# np.savez('train_data.npz', image_names=train_data, labels=train_label)
# print(train_num)

# np.savez('test_data.npz', image_names=test_data, labels=test_label)
# print(len(train_label))
# print(train_data)
# print(train_label)	