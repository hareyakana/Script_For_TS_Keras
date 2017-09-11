from root_pandas import read_root
import numpy as np
import matplotlib.pyplot as plt

def root_single(var,pos):
	# Get the correct size for variables[1] etc.
	array=[]
	output=[]
	variables = var.values
	for x,y in enumerate(variables):
		if y[1] == pos:
			output=(np.append(output,y[0]))

	return output

import keras
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Activation
# from keras.utils.np_utils

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))

dummy1 = np.full(1996,1)
dummy2 = np.zeros(41260,)
dummy = np.concatenate((dummy1,dummy2))
# print(np.shape(dummy))
model = load_model("SimPSDNet.h5")
label = keras.utils.to_categorical(dummy, num_classes=2)

PMTALL = read_root("Training.root","tree",columns=["PSDPara"],flatten=True)
x = PMTALL.PSDPara.values.reshape(43256,31)

# x_abs = abs(x-15200)
# extra = np.arange(4400,31)
# x_mod = np.delete(x_abs,extra,axis=1)

print(x)


# for i in read_root("CAT.root","tree",columns=["PMTALL"],flatten=True,chunksize=size):
# 	x = i.PMTALL
# 	x_np = x.values.reshape(size,4480)

# 	# x_tra = abs(np.transpose(x_np)-16000) #for FFT
# 	x_tra = abs(x_np-15200)
# 	extra = np.arange(4400,4480)
# 	x_trans = np.delete(x_tra,extra,axis=1)

# 	# wave = x.reshape(10,4480)
# 	# print(x_np)

test = read_root("DualGate.root","tree",columns=["PSDPara"],flatten=True)
test = test.PSDPara.values.reshape(1262,31)
# test = abs(test-15200)
# test = np.delete(test,extra,axis=1)

history = model.fit(x,label,epochs=50,batch_size=32, shuffle=True, validation_split=0.1,verbose=1)
scores = model.predict(test[0:100],verbose=1)
eva = model.evaluate(x,label, verbose=1)

# model.save("SimNet.h5")

print(model.metrics_names, scores)
print(history.history.keys())

plt.plot(history.history["loss"])
plt.show()
plt.figure()
plt.plot(history.history["acc"])
plt.show()
plt.figure()
plt.plot(scores)
plt.show()


