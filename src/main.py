import sys
import numpy as np
import tensorflow as tf
import os
import shutil
import cv2
import matplotlib.pyplot as plt
import imutils

from Model import Model, DecoderType
from WordSegmentation import wordSegmentation, prepareImg
from bb import getBoundingBoxes
from database import updateDatabase
from correction import correction

batchSize = 50
imgSize = (128, 32)
maxTextLen = 32

waitKey = 3
divisor = 130

wordSegmentationParameters = {'date_dd': (105, 30, 5, 500), 
								'date_mm': (105, 30, 5, 500), 
								'date_yyyy': (105, 30, 5, 500), 
								'dob_dd': (105, 30, 5, 500), 
								'dob_mm': (105, 30, 5, 500), 
								'dob_yyyy': (105, 30, 5, 500),
								'name': (105, 35, 5, 500), 
								'mobile': (105, 25, 15, 500), 
								'aadhar': (105, 30, 5, 500), 
								'train': (105, 25, 8, 500), 
								'class': (105, 25, 3, 500), 
								'start_stn': (105, 25, 10, 500), 
								'end_stn': (105, 25, 10, 500), 
								'dot_dd': (105, 30, 5, 500), 
								'dot_mm': (105, 30, 5, 500), 
								'dot_yyyy': (105, 30, 5, 500)}

fields = ['Date', 'Name', 'Date_of_Birth',
			'Mobile_Number', 'Aadhar_Number',
			'Train_Number', 'Class',
			'Start_Station', 'End_Station',
			'Date_of_Travel']



# filenames and paths to data
pathCharList = '../model/charList.txt'
pathAccuracy = '../model/accuracy.txt'
pathTrain = '../data/'
pathInfer = '../forms/RailwayForms_8.jpg'
pathCorpus = '../data/corpus.txt'
pathTest = '../test/'

class Batch:
	"batch containing images and ground truth texts"
	def __init__(self, gtTexts, imgs):
		self.imgs = np.stack(imgs, axis=0)
		self.gtTexts = gtTexts


#----------Preprocess----------
def preprocess(img, imgSize, dataAugmentation=False):
	"put img into target img of size imgSize, transpose for TF and normalize gray-values"
	
	# create target image and copy sample image into it
	(wt, ht) = imgSize
	(h, w) = img.shape
	fx = w / wt
	fy = h / ht
	f = max(fx, fy)
	newSize = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1)) # scale according to f (result at least 1 and at most wt or ht)
	img = cv2.resize(img, newSize)
	target = np.ones([ht, wt]) * 255
	target[0:newSize[1], 0:newSize[0]] = img

	# transpose for TF
	img = cv2.transpose(target)

	# normalize
	(m, s) = cv2.meanStdDev(img)
	m = m[0][0]
	s = s[0][0]
	img = img - m
	img = img / s if s>0 else img
	return img


def infer(model, image):
	img = preprocess(image, imgSize)
	batch = Batch(None, [img])
	recognized = model.inferBatch(batch)

	return recognized[0]


def imageSegmentation(model, para):
	#kernelSize, sigma, theta, minArea = para
	recognizedOutput = {}

	numList = ['date_dd', 'date_mm', 'date_yyyy',
	 		'dob_dd', 'dob_mm', 'dob_yyyy',
			'mobile', 'aadhar', 
			'train', 'class',
			'dot_dd', 'dot_mm', 'dot_yyyy']

	# read input images from 'in' directory
	imgFiles = os.listdir('../test/')
	for (i,f) in enumerate(imgFiles):
		imageOutput = []
		images = []
		wordImages = []

		# read
		img = cv2.imread('../test/%s'%f, cv2.IMREAD_GRAYSCALE)

		'''
		# increase contrast
		pxmin = np.min(img)
		pxmax = np.max(img)
		imgContrast = (img - pxmin) / (pxmax - pxmin) * 255

		# increase line width
		kernel = np.ones((2, 2), np.uint8)
		imgMorph = cv2.erode(imgContrast, kernel, iterations = 1)

		# write
		cv2.imwrite('../edited/%s'%f, imgMorph)

		img = cv2.imread('../edited/%s'%f)
		'''
		

		
		
		# read image, prepare it by resizing it to fixed height and converting it to grayscale
		
		img = prepareImg(img, 128)
		f = f.split(".")[0]

		if f == 'sign':
			continue

		(kernelSize, sigma, theta, minArea) = wordSegmentationParameters[f]
		
		# execute segmentation with given parameters
		# -kernelSize: size of filter kernel (odd integer)
		# -sigma: standard deviation of Gaussian function used for filter kernel
		# -theta: approximated width/height ratio of words, filter function is distorted by this factor
		# - minArea: ignore word candidates smaller than specified area
		res = wordSegmentation(img, kernelSize=kernelSize, sigma=sigma, 
									theta=theta, minArea=minArea)
		
		# write output to 'out/inputFileName' directory
		if os.path.exists('../out/%s'%f):
			shutil.rmtree('../out/%s'%f)
			os.mkdir('../out/%s'%f)
		else:
			os.mkdir('../out/%s'%f)
		
		# iterate over all segmented words
		for (j, w) in enumerate(res):
			(wordBox, wordImg) = w
			#print(txt, prob)
			(x, y, w, h) = wordBox
			if h > 40 and w > 40:
				
				txt = infer(model, wordImg)
				imageOutput.append(txt)
				imageOutput.append(" ")
				cv2.imwrite('../out/%s/%d_%s.png'%(f, j, txt), wordImg) # save word
				#print(wordImg.shape)
				#wordImg = preprocess(wordImg, imgSize)
				cv2.rectangle(img,(x,y),(x+w,y+h),0,1) # draw bounding box in summary image
				#wordImages.append(wordImg)
		
		# output summary image with bounding boxes around words
		cv2.imwrite('../out/%s/summary.png'%f, img)
		recognizedOutput[f] = "".join(imageOutput)

	recognizedOutput = correction(recognizedOutput)

	return recognizedOutput

def main():

	# img = tf.keras.preprocessing.image.load_img(pathInfer, target_size=(1170,830))
	# tf.keras.preprocessing.image.save_img(pathInfer, img)
	
	# img = cv2.imread(pathInfer)
	getBoundingBoxes(pathInfer, waitKey, divisor)
	model = Model(charList=open(pathCharList).read(), decoderType=DecoderType.BeamSearch)
	recognizedOutput = imageSegmentation(model, wordSegmentationParameters)


	for i in fields:
		print(i, " -> ", recognizedOutput[i])

	updateDatabase(recognizedOutput)
	# print("Data Updated to the Database...")


if __name__ == '__main__':
	tf.reset_default_graph()
	main()
