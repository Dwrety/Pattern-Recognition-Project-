import numpy as np
import pandas as pd
import keras
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
import os
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import cv2
from subprocess import check_output


df_train = pd.read_csv('../train/labels.csv')
df_test = pd.read_csv('../train/sample_submission.csv')

targets_series = pd.Series(df_train['breed'])
one_hot = pd.get_dummies(targets_series, sparse=True)
one_hot_labels = np.asarray(one_hot)

im_size = 224
# x_train = []
# y_train = []
# x_test = []

# i = 0 
# for img_id, breed in tqdm(df_train.values):
# 	img = cv2.imread('../train/{}.jpg'.format(img_id))
# 	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 	label = one_hot_labels[i]
# 	x_train.append(cv2.resize(img, (im_size, im_size)))
# 	y_train.append(label)
# 	i += 1

# for img_id in tqdm(df_test['id'].values):
# 	img = cv2.imread('../test/{}.jpg'.format(img_id))
# 	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 	x_test.append(cv2.resize(img, (im_size, im_size)))

# y_train_raw = np.array(y_train, np.uint8)
# x_train_raw = np.array(x_train, np.float32)/255.
# x_test  = np.array(x_test, np.float32)/255.
# np.savez("train_pack.npz", x_train_raw=x_train_raw,y_train_raw=y_train_raw,x_test=x_test)

y_train_raw = np.load('train_pack.npz')['y_train_raw']
x_train_raw = np.load('train_pack.npz')['x_train_raw']
x_test = np.load('train_pack.npz')['x_test']

# print(x_train_raw.shape)
# print(y_train_raw.shape)
# print(x_test.shape)

num_class = y_train_raw.shape[1]
X_train, X_valid, Y_train, Y_valid = train_test_split(x_train_raw, y_train_raw, test_size=0.3, random_state=1)

base_model= VGG19(weights='imagenet', include_top=False, input_shape=(im_size, im_size, 3))
x = base_model.output
x = Flatten()(x)
predictions = Dense(num_class, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers:
	layer.trainable = False

model.compile(loss='categorical_crossentropy',
				optimizer='adam',
				metrics=['accuracy'])
callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=1)]
model.fit(X_train, Y_train, epochs=10, validation_data=(X_valid, Y_valid), verbose=1,shuffle=True)
# preds = model.predict(x_test, verbose=1)
