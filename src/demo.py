import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image


img = cv2.imread('../test/test.jpg', cv2.IMREAD_GRAYSCALE)
img = np.array(img)

print(img.shape)
for i in range(img.shape[0]):
	for j in range(img.shape[1]):
		if img[i][j] < 255:
			img[i][j] = 0


img_1 = Image.fromarray(img)

imgplot = plt.imshow(img_1)
plt.show()


