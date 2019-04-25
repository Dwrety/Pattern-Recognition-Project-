import numpy as np
import util
import matplotlib.pyplot as plt
import visual_words
import skimage.color
import visual_recog
import skimage.io
import cv2


if __name__ == '__main__':

	num_cores = util.get_num_CPU()

	path_img = "train/bernese_mountain_dog/0c36c19e7c4e932b8e0c01aa845b2fce.jpg"
	image = cv2.imread(path_img)
	img_correct = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
	img = cv2.imread('sift_keypoints.jpg')
	plt_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	fig = plt.figure(1)
	plt.subplot(121)
	plt.imshow(img_correct)
	plt.axis("off")
	plt.subplot(122)
	plt.imshow(plt_img)
	plt.axis("off")
	plt.show()
	# image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

	# print(image_gray)
	# sift = cv2.xfeatures2d.SIFT_create()
	# kp = sift.detect(image_gray,None)
	# img= cv2.drawKeypoints(image_gray,kp,image,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	# cv2.imwrite('sift_keypoints.jpg',img)
	# for point in kp:
	# 	print(point.pt)

	image = image.astype('float')/255.
	filter_responses = visual_words.extract_filter_responses(image)
	fff = filter_responses.reshape(-1, filter_responses.shape[2])
	# print(fff.shape)
	util.display_filter_responses(filter_responses)



	#visual_words.compute_dictionary(num_workers=num_cores)
	
	#dictionary = np.load('dictionary.npy')
	#img = visual_words.get_visual_words(image,dictionary)
	#util.save_wordmap(wordmap, filename)
	#visual_recog.build_recognition_system(num_workers=num_cores)

	#conf, accuracy = visual_recog.evaluate_recognition_system(num_workers=num_cores)
	#print(conf)
	#print(np.diag(conf).sum()/conf.sum())

	#vgg16 = torchvision.models.vgg16(pretrained=True).double()
	#vgg16.eval()
	#deep_recog.build_recognition_system(vgg16,num_workers=num_cores//2)
	#conf = deep_recog.evaluate_recognition_system(vgg16,num_workers=num_cores//2)
	#print(conf)
	#print(np.diag(conf).sum()/conf.sum())

