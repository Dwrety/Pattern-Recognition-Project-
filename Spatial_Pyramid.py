# import packages here
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.misc import imresize  # resize images
import copy
from sklearn.cluster import KMeans
from sklearn import preprocessing
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from mpl_toolkits.axes_grid1 import ImageGrid


# class_names = [name[6:] for name in glob.glob('train/*')]
# class_names = dict(zip(range(0,len(class_names)), class_names))
# # print(class_names)
# NUM_CLASSES =24

dictionary = np.load("dictionary.npy")
# dictionary = dictionary.reshape(200,20,3)
plt.imshow(dictionary.T+np.abs(np.min(dictionary)), cmap='gray')
plt.show()
# for key in dictionary:
# 	# f = np.zeros((4,5,3))
# 	plt.imshow(key)
# 	# f[:,:,0] = key[0:20].reshape(4,5)
# 	# f[:,:,1] = key[20:40].reshape(4,5)
# 	# f[:,:,2] = key[40:60].reshape(4,5)
# 	# plt.imshow(f, cmap='gray')
# 	plt.show()



def load_dataset(path, num_per_class=-1):
	data = []
	labels = []
	for idx, class_name in class_names.items():
		img_path_class = glob.glob(path + class_name + '/*.jpg')
		# print(img_path_class)
		if num_per_class > 0:
			img_path_class = img_path_class[:num_per_class]
		labels.extend([idx]*len(img_path_class))
		for filename in img_path_class:
			data.append(filename)
	labels = np.array(labels)
	data = np.array(data)
	return data, labels


# # load training dataset
# train_data, train_label = load_dataset('train/', 140)
# train_num = len(train_label)
# np.savez('train_data.npz', image_names=train_data, labels=train_label)
# # print(train_num)
# test_data, test_label = load_dataset('test/', 100)
# np.savez('test_data.npz', image_names=test_data, labels=test_label)
# print(len(train_label))
# print(train_data)
# print(train_label)

# take one picture from each class
# print(class_names)
# wtf = []

# for idx, class_name in class_names.items():
# 	path = 'train/'
# 	img_path_class = glob.glob(path + class_name + '/*.jpg')
# 	i = np.random.randint(0, len(img_path_class))
# 	img = cv2.imread(img_path_class[i])
# 	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 	img = cv2.resize(img,(224,224))
# 	wtf.append((img, idx))

# i = int(np.sqrt(NUM_CLASSES))
# j = int(np.ceil(1. * NUM_CLASSES / i))
# print(i,j)
# fig = plt.figure(1, figsize=(20, 20))
# grid = ImageGrid(fig, 111, nrows_ncols=(i, j), axes_pad=0.05)
# for i, (img,c) in enumerate(wtf):
# 	ax = grid[i]
# 	print(class_names[c])
# 	ax.imshow(img / 255.)
# 	ax.text(10, 210, 'LABEL: %s' % class_names[c], color='k', backgroundcolor='w', fontsize=8)
# 	ax.axis('off')
# plt.show()


# # feature extraction
# def extract_feat(raw_data):
# 	num_sample = len(raw_data)
# 	feat_dim = 4096
# 	feat = np.zeros((num_sample, feat_dim), dtype=np.float32)
# 	for i in range(num_sample):
# 		feat[i] = raw_data[i].flatten()[:feat_dim] # dummy implemtation
# 	return feat

# train_feat = extract_feat(train_data)
# print(train_feat.shape)
# test_feat = extract_feat(test_data)
# print(test_feat.shape)

# def train(X, Y):
# 	return 0

# def predict(model, x):
# 	return np.random.randint(24) # dummy implementation	

# # evaluation
# predictions = [-1]*len(test_feat)
# # print(predictions)

# for i in range(0, test_num):
# 	predictions[i] = predict(None, test_feat[i])
# accuracy = sum(np.array(predictions) == test_label) / float(test_num)
# print ("The accuracy of my dummy model is {:.2f}%".format(accuracy*100))


# compute dense SIFT
def computeSIFT(data, draw=False):
	x = []
	num_data = len(data)
	for i in tqdm(range(num_data)):
		sift = cv2.xfeatures2d.SIFT_create()
		img = data[i]
		# print(img)
		step_size = 20
		kp = [cv2.KeyPoint(x, y, step_size) for x in range(0, img.shape[1], step_size) for y in range(0, img.shape[0], step_size)]
		# kp = sift.detect(img, None)
		dense_feat = sift.compute(img, kp)
		# kp, des = sift.detectAndCompute(img, None)
		if draw:
			res = cv2.drawKeypoints(img, kp)
			cv2.imwrite('sift_keypoints.jpg',res)
		x.append(dense_feat[1])
	return x


# extract dense sift features from training images
# x_train = computeSIFT(train_data)
# x_test = computeSIFT(test_data)
# img_0 = test_data[5]
# ggg = computeSIFT([img_0], draw=True)

# all_train_desc = []
# for i in range(len(x_train)):
# 	for j in range(x_train[i].shape[0]):
# 		all_train_desc.append(x_train[i][j,:])
# all_train_desc = np.array(all_train_desc)

# train model
def trainKNN(data, labels, k):
    neigh = KNeighborsClassifier(n_neighbors=k, p=2)
    neigh.fit(data, labels) 
    return neigh

def clusterFeatures(all_train_desc, k):
	kmeans = KMeans(n_clusters=k,random_state=0).fit(all_train_desc)
	return kmeans

# form training set histograms for each training image using BoW representation
def formTrainingSetHistogram(x_train, kmeans, k):
	train_hist = []
	for i in range(len(x_train)):
		data = copy.deepcopy(x_train[i])
		predict = kmeans.predict(data)
		train_hist.append(np.bincount(predict, minlength=k).reshape(1,-1).ravel())
	return np.array(train_hist)

# build histograms for test set and predict
def predictKMeans(kmeans, scaler, x_test, train_hist, train_label, k):
	# form histograms for test set as test data
	test_hist = formTrainingSetHistogram(x_test, kmeans, k)

	# make testing histograms zero mean and unit variance
	test_hist = scaler.transform(test_hist)

	# Train model using KNN
	knn = trainKNN(train_hist, train_label, k)
	predict = knn.predict(test_hist)
	return np.array([predict], dtype=np.array([test_label]).dtype)


def accuracy(predict_label, test_label):
	return np.mean(np.array(predict_label.tolist()[0]) == np.array(test_label))


# k = [100]
# for i in range(len(k)):
# 	kmeans = clusterFeatures(all_train_desc, k[i])
# 	train_hist = formTrainingSetHistogram(x_train, kmeans, k[i])

# 	# preprocess training histograms
# 	scaler = preprocessing.StandardScaler().fit(train_hist)
# 	train_hist = scaler.transform(train_hist)

# 	predict = predictKMeans(kmeans, scaler, x_test, train_hist, train_label, k[i])
# 	res = accuracy(predict, test_label)
# 	print("k =", k[i], ", Accuracy:", res*100, "%")