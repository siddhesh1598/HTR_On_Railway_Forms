import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import random


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


def correctnboxes(nboxes):
    if nboxes[2][0] > nboxes[3][0]:
        nboxes[2], nboxes[3] = nboxes[3], nboxes[2]
    if nboxes[6][0] > nboxes[7][0]:
        nboxes[6], nboxes[7] = nboxes[7], nboxes[6]
    if nboxes[8][0] > nboxes[9][0]:
        nboxes[8], nboxes[9] = nboxes[9], nboxes[8]
    if nboxes[10][0] > nboxes[11][0]:
        nboxes[10], nboxes[11] = nboxes[11], nboxes[10]
    if nboxes[12][0] > nboxes[13][0]:
        nboxes[12], nboxes[13] = nboxes[13], nboxes[12]


def getBoundingBoxes(img_path, waitKey):
	divisor = 80

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
	    if (w > 60 and h > 20) and w > 3*h:
	        boxes.append([x, y, x+w, y+h])

	nboxes = []
	for i in boxes:
	    flag = 1
	    x1i, y1i, x2i, y2i = i
	    for j in boxes:
	        x1j, y1j, x2j, y2j = j
	        if x1i<x1j<x2i and y1i<y1j<y2i:
	            flag = 0

	    if flag:
	        nboxes.append(i)

	correctnboxes(nboxes)

	image = cv2.imread(img_path)
	image = imutils.resize(image)
	title = ['date', 'name', 'date_of_birth', 'aadhar_no', 'address_line_1',
	        'address_line_2', 'address_line_3', 'pincode', 'mobile_no',
	        'phone_no', 'train_no', 'class', 'start_station', 'end_station',
	        'date_of_travel']
	count = 0
	for i in nboxes:
		x1, y1, x2, y2 = i
		cv2.rectangle(image, (x1,y1), (x2,y2), 
							(100,100,100), 2)
		crop_img = image[y1:y2, x1:x2]

		if count <= 14:
			cv2.imwrite('../test/' + title[count] + '.jpg', crop_img)
			cv2.putText(image, title[count], (x1, y1-2), cv2.FONT_ITALIC, 0.55, 
						(100,100,255), 1)
		else:
			break

		count += 1

	cv2.imshow("image", image)
	cv2.waitKey(waitKey*1000)
	cv2.destroyAllWindows()


#getBoundingBoxes('../forms/form.jpg', 3)