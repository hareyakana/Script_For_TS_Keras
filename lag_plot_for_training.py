import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import misc

# image = img.imread("image_test/000_diff_lag.png")
image = misc.imread("image_test/000_diff_lag.png")

print(image.shape)

grey = np.zeros((image.shape[0],image.shape[1]))

for i in range(len(image)):
	for j in range(len(image[i])):
		grey[i][j] = np.average(image[i][j])
		# print(grey[i][j])
		# print(j)

def cropping(image):
	non_x1 = np.arange(0, 60)
	non_x2 = np.arange(540, 600)
	non_y1 = np.arange(0, 100)
	non_y2 = np.arange(700, 800)

	image = np.delete(image, non_x2, axis=0)
	image = np.delete(image, non_x1, axis=0)
	image = np.delete(image, non_y2, axis=1)
	image = np.delete(image, non_y1, axis=1)
	return image

print(cropping(grey))
plt.imshow(cropping(grey), cmap=cm.Greys_r)
plt.show()