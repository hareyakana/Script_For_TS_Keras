import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import misc
from pandas import Series

from root_pandas import read_root

pmtall = read_root("RefPulse/RefPulse.root","tree",columns=["PMTALL"],flatten=True)

from sklearn.preprocessing import normalize

def NORMPMTALL(pmtall,num_entries):
	"""
	Convert read_root into numpy for speed
	invert the PMT pulse
	get rid of the last 80 bin as those bins of the pulse do not matter
	normalize all waveform to rid of energy dependency 
	"""
	x = pmtall.PMTALL.values.reshape(num_entries,4480)
	x_abs = abs(x-15200)
	extra = np.arange(4400,4480)
	x_mod = np.delete(x_abs,extra,axis=1)
	x_nor = normalize(x_mod,norm="l2")
	# print(np.sum(x_nor,axis=1))
	return x_nor

PMTALL = NORMPMTALL(pmtall,1996)

from pandas.plotting import lag_plot, autocorrelation_plot

# number of events to analyse
n = 10

events = np.zeros((n,640,480))

for i in range(n):
	# load data
	wave = Series(PMTALL[i,:])
	# lag plot of the single event (time series)
	fig = plt.figure() 
	lag_plot(wave)
	# convert plot into numpy array(to feed into CNN later)
	fig.canvas.draw()
	w,h = fig.canvas.get_width_height()
	pltarray = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
	pltarray.shape = (w,h,3)
	plt.close() #delete plot to save memory
	# convert rgb into greyscale since colors do not really matter here. onl two values concerned
	array = np.zeros((w,h))
	for one in range(len(pltarray)):
		for two in range(len(pltarray[one])):
			array[one][two] = np.average(pltarray[one][two])

	events[i] = array

	# fig1=plt.figure()
	# plt.imshow(array,cmap=cm.Greys_r)
	# plt.show(fig1)
	# print(np.shape(grey))
	# plt.imshow(grey)
	# plt.close()

print(events[1])
print(np.shape(events))
print(np.shape(events[1]))

# 

# image = misc.imread("image_test/000_diff_lag.png")

# print(image.shape)

# grey = np.zeros((image.shape[0],image.shape[1]))

# for i in range(len(image)):
# 	for j in range(len(image[i])):
# 		grey[i][j] = np.average(image[i][j])

# """ 
# 480 x 600 input of single lagplot image
# """

# def cropping(image):
# 	non_x1 = np.arange(0, 60)
# 	non_x2 = np.arange(540, 600)
# 	non_y1 = np.arange(0, 100)
# 	non_y2 = np.arange(720, 800)

# 	image = np.delete(image, non_x2, axis=0)
# 	image = np.delete(image, non_x1, axis=0)
# 	image = np.delete(image, non_y2, axis=1)
# 	image = np.delete(image, non_y1, axis=1)
# 	return image

# print(cropping(grey))
# plt.imshow(cropping(grey), cmap=cm.Greys_r)
# plt.show()