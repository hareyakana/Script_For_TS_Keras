import numpy as np
import matplotlib.pyplot as plt

"""
MISSION IMPOSSIBLE of classification of gamma and beta, fix y1 to counting rather than just straight up shifting.

IDEAL SITUATION: a simulation program capable of simulating pulse shape as this we have perfect information about the physics that generates it.
from here we can generate the data set required to construct the classifier program
in turn we can augment our input data that is suitable for CNN network to classify general events with goals of achieving distinguishtion between gamma and beta in a detector designed not to separately identify it.

However we do not have the tools available so we have to dummfied our concept with a generic fucntion which this script set out to do.

POSSIBLE input: ARIMA parameters, gram matrix, markov transition matrix, lag plot, markov transition "field".

1. create a false pulse generator that has the general shape
2. add noise
3. use one of the augmentation method on the false data


"""
Ratio = np.random.rand(3)
Ratio = Ratio/sum(Ratio)

x = np.arange(0,1,0.01)
y = np.exp(-x)

# y1 = Ratio[0]*np.exp(-(x+0.002*x))+Ratio[1]*np.exp(-(x+x*0.007))+Ratio[2]*np.exp(-x)
y1 = np.zeros(len(x))
# for i in range(len(x)):
# 	if i <int(0.5*len(x)):
# 		y1[i] = np.exp(-2*x[i])
# 	if i>= int(0.5*len(x)):
# 		y1[i] = np.exp(-(x[i]+0.5))

gap = np.random.rand(1)*0.2
for i in range(len(x)):
	if i <int(gap*len(x)):
		y1[i] = 0.01
	if i>= int(gap*len(x)):
		y1[i] = np.exp(-x[i])

fig1 = plt.figure()
plt.plot(x,y,color="r")
plt.plot(x,y1,color="y")
# plt.xlim(0.42,0.48)
# plt.ylim(0.62,0.66)
plt.show()
# fig2 = plt.figure()
# plt.plot(x,y,color="r")
# plt.plot(x,y1,color="y")
# plt.show()

"""
add noise
"""
noise = 0.5

y_noise = (np.random.rand(len(x))*noise-noise/2.)+y
y1_noise = (np.random.rand(len(x))*noise-noise/2.)+y1

# fig3 = plt.figure()
# plt.plot(x,y_noise,color="r")
# plt.plot(x,y1_noise,color="y")
# plt.xlim(0.42,0.48)
# plt.ylim(0.62,0.66)
# plt.show()
# fig4 = plt.figure()
# plt.plot(x,y_noise,color="r")
# plt.plot(x,y1_noise,color="y")
# plt.show()

# print(y_noise*0.01)

"""
Personal suggestion
create a set of false waveform pulse of beta, try to distinguish in from y and y1

add random generator to the ratio of gamma pulses
"""
n = 2000
no_of_x = 100

def x_generator(n):
	input_x = np.zeros((n,no_of_x))
	for i in range(n):
		input_x[i] = np.arange(0,1,0.01)
	return input_x

x_axis = x_generator(n)

def beta(x):
	yall = np.zeros((n,no_of_x))
	for i in range(n):
		yall[i] = np.exp(-x[i]) + (np.random.rand(no_of_x)*noise - noise/2.)

	return yall

# print(beta(x_axis))

def gamma(x):
	yall = np.zeros((n,no_of_x))
	for i in range(n):
		# ratio = np.random.rand(3)
		# ratio = ratio/sum(ratio)
		# shift = np.random.rand(2)/no_of_x
		# yall[i] = ratio[0]*np.exp(-(x[i]+shift[0]*x[i])) + ratio[1]*np.exp(-(x[i]+shift[1]*x[i])) + ratio[2]*np.exp(-x[i])
		# yall[i] = yall[i] + (np.random.rand(no_of_x)*noise - noise/2.)
		# yall[i] = np.exp(-2*x[i]) + (np.random.rand(no_of_x)*noise - noise/2.)

	# """ Piece wise function"""
		# for j in range(no_of_x):
		# 	if j < int(0.5*no_of_x):
		# 		yall[i][j] = np.exp(-2*x[i][j]) + (np.random.rand(1)*noise - noise/2.)
		# 	if j >= int(0.5*no_of_x):
		# 		yall[i][j] = np.exp(-(x[i][j]+0.5)) + (np.random.rand(1)*noise - noise/2.)

		pileup = np.random.rand(1)*0.3
		for j in range(no_of_x):
			if j < int(pileup*no_of_x):
				yall[i][j] = 0.01 + 0.01*(np.random.rand(1)*noise - noise/2.)
			if j >= int(pileup*no_of_x):
				yall[i][j] = np.exp(-x[i][j]) + (np.random.rand(1)*noise - noise/2.)

	return yall

# print(gamma(x_axis))

fig5 = plt.figure()
plt.plot(x_axis[1],beta(x_axis)[1])
plt.plot(x_axis[1],gamma(x_axis)[1])
plt.show()



# Ratio = np.random.rand(3)
# Ratio = Ratio/sum(Ratio)

# Y = np.exp(-x) + (np.random.rand(len(x))*0.01-0.005)
# Y1 = Ratio[0]*np.exp(-(x+np.random.rand(1)*0.002*x))+Ratio[1]*np.exp(-(x+np.random.rand(1)*x*0.006))+Ratio[2]*np.exp(-x) + (np.random.rand(len(x))*0.01-0.005)

# fig4 = plt.figure()
# plt.bar(x,Y,0.01,color="r",alpha=0.9)
# plt.bar(x,Y1,0.01,color="y",alpha=0.9)
# plt.xlim(0.42,0.48)
# plt.ylim(0.62,0.66)
# plt.show()


"""
Produce a set to randomly generated pulse shape based on the general pulse shape of beta and gamma
""" 


"""
Augmentation of time series - GRAMMIAN
"""
import math
def resecale_1to1(pmtall):
	x = pmtall
	w,h = x.shape
	print(w,h)
	y = np.zeros((w,h))
	x_max = np.amax(x, axis = 1)
	x_min = np.amin(x, axis = 1)
	for i in range(len(x)):
		for j in range(len(x[i])):
			y0 = (x[i][j])
			y1 = x_min[i]
			y2 = x_max[i]
			y[i][j] = ((y0-y2)+(y0 - y1))/(y2 - y1)
	return y


def polar(pmtall):
	x = pmtall
	w,h = x.shape
	print("polar!!")
	r = np.zeros((w,h))
	# phi = np.zeros((w,h))
	for i in range(len(x)):
		j = len(x[i])
		for k in range(len(x[i])):
			y = x[i][k]
			r[i][k] = math.acos(y)*2 - math.pi
			# phi[i][k] = k/j

	return r

def gram_matrix(pmtall):
	x = pmtall
	w,h = x.shape
	print("gramian!")
	y = np.zeros((w,h,h))
	for i in range(w):
		for j in range(h):
			for k in range(h):
				ele1 = x[i][j]
				ele2 = x[i][k]
				y[i][j][k] = math.sin(ele1+ele2)
		# plt.imshow(y[i], interpolation="nearest")
		# plt.show()

	return y
x1 = beta(x_axis)
x2 = gamma(x_axis)
x1_1 = resecale_1to1(x1)
x2_1 = resecale_1to1(x2)
polar_coor_beta = polar(x1_1)
polar_coor_gamma = polar(x2_1)
gram_beta = gram_matrix(polar_coor_beta)
gram_gamma = gram_matrix(polar_coor_gamma)

print(gram_beta.shape)

"""
Augmentation of Time Series - STOCHASTIC Matrix
"""
from sklearn.preprocessing import normalize

def stochastic(x):
	w,h = x.shape
	Matrix = np.zeros((w,h,h))
	for i in range(w):
		bins = np.arange(0, 1, 0.01)
		inds = np.digitize(x[i], bins, right=True)

		lags = np.zeros((2,len(inds)-1))
		for y in range(len(inds)-1):
			lags[0][y] = inds[y]
			lags[1][y] = inds[y+1]

		MTM = np.zeros((len(bins),len(bins)))
		for y in range(len(bins)):
			for z in range(len(bins)):

				dummy = 0
				for p in range(len(lags[0])):
					if lags[0][p] == y and lags[1][p] == z :
						dummy += 1
				MTM[y][z] = dummy
		MTM = normalize(MTM,norm="l1")
		MTM = np.flipud(MTM)
		Matrix[i] = MTM
	return Matrix

# markov_beta = stochastic(x1)
# markov_gamma = stochastic(x2)

# print(markov_beta.shape)

"""
Prepare input data for Keras
"""

def separation(x,ratio):
	train = np.zeros((int(len(x)*ratio),no_of_x,no_of_x))
	test = np.zeros((int(len(x)*(1.-ratio)),no_of_x,no_of_x))
	for i in range(n):
		if i < (n*ratio):
			train[i] = x[i]
		if i >= (n*ratio):
			k = i - int(n*ratio) - 1
			test[k] = x[i]
	return train, test

train_beta, test_beta = separation(gram_beta,0.5)
train_gamma, test_gamma  = separation(gram_gamma,0.5)

# train_beta, test_beta = separation(markov_beta,0.5)
# train_gamma, test_gamma  = separation(markov_gamma,0.5)

def labelling(x,num):
	label = np.zeros(len(x))
	for i in range(len(x)):
		label[i] = num
	return label

label_train_beta = labelling(train_beta,1)
label_train_gamma = labelling(train_gamma,0)
label_test_beta = labelling(test_beta,1)
label_test_gamma = labelling(test_gamma,0)

data_train = np.concatenate((train_beta,train_gamma), axis=0)
data_test = np.concatenate((test_beta,test_gamma), axis=0)
label_train = np.concatenate((label_train_beta,label_train_gamma),axis=0)
label_test = np.concatenate((label_test_beta,label_test_gamma),axis=0)

"""
Save output for debugging!
"""

np.save("data_train.npy",data_train)
np.save("data_test.npy",data_test)
np.save("label_train.npy",label_train)
np.save("label_test.npy",label_test)

data_train = np.load("data_train.npy")
data_test = np.load("data_test.npy")
label_train = np.load("label_train.npy")
label_test = np.load("label_test.npy")

data_train = np.expand_dims(data_train,axis=3)
data_test = np.expand_dims(data_test,axis=3)

"""
training program 
"""

import keras

label_train = keras.utils.to_categorical(label_train, 2)
label_test = keras.utils.to_categorical(label_test, 2)

from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization

# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# set_session(tf.Session(config=config))

from keras.regularizers import l2
from keras.optimizers import SGD

""" Dense Network """

# model = Sequential()
# model.add(Dense(100, activation="relu", input_shape=(100,100,1)))
# model.add(Flatten())
# model.add(Dense(200, activation="relu"))
# model.add(Dense(300, activation="relu"))
# model.add(Dense(2,activation="softmax"))
# model.summary()
# model.compile(loss = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Adadelta(), metrics=['accuracy'])
# model.fit(data_train, label_train, batch_size=128, epochs=12, verbose=1, validation_data=(data_test,label_test))
# score = model.evaluate(data_test, label_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

""" Convulution Neural Network """

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation="relu", input_shape=(100,100,1)))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation="relu"))
# model.add(Dense(256,activation="relu"))
# model.add(Dense(512,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(2,activation="softmax"))
model.summary()
model.compile(loss = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Adadelta(), metrics=['accuracy'])
model.fit(data_train, label_train, batch_size=50, epochs=10, verbose=1, validation_data=(data_test,label_test))
score = model.evaluate(data_test, label_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

""" ResNet50 attempt , needs debugging!"""

# def identity_block(input_tensor, kernel_size, filters, stage, block):

# 	conv_name_base = 'res' + str(stage) + block + '_branch'
# 	bn_name_base = 'bn' + str(stage) + block + "_branch"

# 	x = Conv2D(filters, (1, 1), name=conv_name_base + '2a')(input_tensor)
# 	x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
# 	x = Activation('relu')(x)

# 	x = Conv2D(filters, kernel_size, padding='same', name=conv_name_base + '2b')(x)
# 	x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
# 	x = Activation('relu')(x)

# 	x = Conv2D(filters, (1, 1), name=conv_name_base + '2c')(x)
# 	x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

# 	x = layers.add([x, input_tensor])
# 	x = Activation('relu')(x)

# 	return x

# def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):

# 	conv_name_base = 'res' + str(stage) + block + '_branch'
# 	bn_name_base = 'bn' + str(stage) + block + '_branch'
# 	print(input_tensor)
# 	x = Conv2D(filters, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor) #
# 	x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
# 	x = Activation('relu')(x)

# 	x = Conv2D(filters, kernel_size, padding='same', name=conv_name_base + '2b')(x)
# 	x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
# 	x = Activation('relu')(x)

# 	x = Conv2D(filters, (1, 1), name=conv_name_base + '2c')(x)
# 	x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

# 	shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
# 	shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(shortcut)

# 	x = layers.add([x, shortcut])
# 	x = Activation('relu')(x)

# 	return x

# def ResNet50(input_tensor, classes):

# 	x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(input_tensor)
# 	x = BatchNormalization(axis=3, name='bn_conv1')(x)
# 	x = Activation('relu')(x)
# 	x = MaxPooling2D((3, 3), strides=(2, 2))(x)
# 	print(x)

# 	x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1)) #
# 	x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
# 	x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

# 	x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
# 	x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
# 	x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
# 	x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

# 	x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
# 	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
# 	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
# 	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
# 	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
# 	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

# 	x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
# 	x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
# 	x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

# 	x = AveragePooling2D((7, 7), name='avg_pool')(x)

# 	x = Flatten()(x)
# 	x = Dense(classes, activation='softmax', name='fc2')(x)

# 	# x = GlobalMaxPooling2D()(x)

# 	model = Model(input_tensor, x, name="resnet50")

# 	return model

# ResNet50(input_tensor=Input(shape=(100,100,1)), classes=2)











