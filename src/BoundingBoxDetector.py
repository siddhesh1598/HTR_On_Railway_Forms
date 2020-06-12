import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import random

def area(box):
	x1, y1, x2, y2 = box
	w = abs(y2 - y1)
	h = abs(x2 - x1)

	return w*h

def iou(box1, box2):
	"""Implement the intersection over union (IoU) between box1 and box2

	Arguments:
	box1 -- first box, list object with coordinates (x1, y1, x2, y2)
	box2 -- second box, list object with coordinates (x1, y1, x2, y2)
	"""

	if box1[3] < box2[1]:
		return 0
	else:
		# Calculate the (y1, x1, y2, x2) coordinates of the intersection of box1 and box2. Calculate its Area.
		xi1 = np.maximum(box1[0], box2[0])
		yi1 = np.maximum(box1[1], box2[1])
		xi2 = np.minimum(box1[2], box2[2])
		yi2 = np.minimum(box1[3], box2[3])
		inter_area = (xi2-xi1)*(yi2-yi1)  

		# Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
		box1_area = (box1[2]-box1[0])*(box1[3]-box1[1])
		box2_area = (box2[2]-box2[0])*(box2[3]-box2[1])
		union_area = box1_area + box2_area - inter_area

		# compute the IoU
		iou = inter_area/union_area

		return iou

def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0

	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True

	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1

	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))

	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)

def refinedBoxes(boxes):
	nboxes = []
	visited = [0] * len(boxes)

	for i in boxes:
		if visited[boxes.index(i)] == 0:
			box = i
			visited[boxes.index(i)] = 1

			for j in range(boxes.index(i)+1, len(boxes)):
				if iou(box, boxes[j]) > 0.30 and visited[j] == 0:
					visited[j] = 1
					if area(boxes[j]) < area(box):
						box = boxes[j]

			nboxes.append(box)
	
	x_min = min(nboxes[0][0], nboxes[1][0], nboxes[2][0])
	x_max = max(nboxes[0][0], nboxes[1][0], nboxes[2][0])

	if nboxes[1][0] == x_min:
		nboxes[0], nboxes[1] = nboxes[1], nboxes[0]
	elif nboxes[2][0] == x_min:
		nboxes[0], nboxes[2] = nboxes[2], nboxes[0]
	if nboxes[0][0] == x_max:
		nboxes[0], nboxes[2] = nboxes[2], nboxes[0]
	elif nboxes[1][0] == x_max:
		nboxes[1], nboxes[2] = nboxes[2], nboxes[1]


	x_min = min(nboxes[3][0], nboxes[4][0], nboxes[5][0], nboxes[6][0])
	x_max = max(nboxes[3][0], nboxes[4][0], nboxes[5][0], nboxes[6][0])

	if nboxes[4][0] == x_min:
		nboxes[3], nboxes[4] = nboxes[4], nboxes[3]
	elif nboxes[5][0] == x_min:
		nboxes[3], nboxes[5] = nboxes[5], nboxes[3]
	elif nboxes[6][0] == x_min:
		nboxes[3], nboxes[6] = nboxes[6], nboxes[3]
	if nboxes[3][0] == x_max:
		nboxes[3], nboxes[6] = nboxes[6], nboxes[3]
	elif nboxes[4][0] == x_max:
		nboxes[4], nboxes[6] = nboxes[6], nboxes[4]
	elif nboxes[5][0] == x_max:
		nboxes[5], nboxes[6] = nboxes[6], nboxes[5]


	if nboxes[4][0] > nboxes[5][0]:
		nboxes[4], nboxes[5] = nboxes[5], nboxes[4]

	if nboxes[7][0] > nboxes[8][0]:
		nboxes[7], nboxes[8] = nboxes[8], nboxes[7]
	if nboxes[9][0] > nboxes[10][0]:
		nboxes[9], nboxes[10] = nboxes[10], nboxes[9]
	if nboxes[12][0] > nboxes[13][0]:
		nboxes[12], nboxes[13] = nboxes[13], nboxes[12]

	x_min = min(nboxes[14][0], nboxes[15][0],nboxes[16][0])
	x_max = max(nboxes[14][0], nboxes[15][0],nboxes[16][0])

	if nboxes[15][0] == x_min:
		nboxes[14], nboxes[15] = nboxes[15], nboxes[14]
	elif nboxes[16][0] == x_min:
		nboxes[14], nboxes[16] = nboxes[16], nboxes[14]
	if nboxes[14][0] == x_max:
		nboxes[14], nboxes[16] = nboxes[16], nboxes[14]
	elif nboxes[15][0] == x_max:
		nboxes[15], nboxes[16] = nboxes[16], nboxes[15]

	return nboxes
	

def getBoundingBoxes(img_path, waitKey, divisor):
	divisor = divisor

	# Read the image
	img = cv2.imread(img_path, 0)
	img = imutils.resize(img)

	# Thresholding the image
	(thresh, img_bin) = cv2.threshold(img, 128, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	# Invert the image
	img_bin = 255-img_bin

	# Defining a kernel length
	kernel_length = np.array(img).shape[1]//divisor

	# A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
	verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))

	# A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
	hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))

	# A kernel of (3 X 3) ones.
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

	# Morphological operation to detect vertical lines from an image
	img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
	verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)

	# Morphological operation to detect horizontal lines from an image
	img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
	horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)

	# Weighting parameters, this will decide the quantity of an image to be added to make a new image.
	alpha = 0.5
	beta = 1.0 - alpha

	# This function helps to add two image with specific weight parameter to get a third image as summation of two image.
	img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
	img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
	(thresh, img_final_bin) = cv2.threshold(img_final_bin, 128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


	# Find contours for image, which will detect all the boxes
	(contours, _) = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


	# Sort all the contours by top to bottom.
	(contours, boundingBoxes) = sort_contours(contours, method="top-to-bottom")

	boxes = []
	for c in contours:
		# Returns the location and width,height for every contour
		x, y, w, h = cv2.boundingRect(c)
		if (w > 50 and h > 30) and h < 300:
			boxes.append([x, y, x+w, y+h])

	# print(boxes)
	boxes = refinedBoxes(boxes)
	image = cv2.imread(img_path, 1)

	title = ['date_dd', 'date_mm', 'date_yyyy', 
			'name', 'dob_dd', 'dob_mm', 'dob_yyyy',
			'mobile', 'aadhar', 
			'train', 'class', 
			'start_stn', 
			'end_stn', 'sign',
			'dot_dd', 'dot_mm', 'dot_yyyy']

	count = 0
	for i in boxes:
		x1, y1, x2, y2 = i
		cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 5)
		crop_img = img[y1:y2, x1:x2]
		cv2.imwrite('../test/' + title[count] +  '.jpg', crop_img)
		cv2.putText(image, title[count], (x1, y1-2), cv2.FONT_ITALIC, 0.55, 
						(100,100,255), 1)
		count += 1
		if count == 17:
			break


	cv2.imwrite("image.jpg", image)

	'''
	for i in range(len(boxes)):
		if i == 0 or i == 25 or i == 26:
			x1, y1, x2, y2 = boxes[i]
			cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 3)

	#cv2.imwrite("image.jpg", img)
	print(boxes[0])
	print(boxes[25])
	print(boxes[26])
	'''

# getBoundingBoxes('../forms/RailwayForms_5.jpg', 10, 130)
    
                