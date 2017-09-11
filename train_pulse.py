# Same as 2.7train.py but for pulse shape instead
# import keras.backend as K

# def mean_pred(y_true, y_pred):
#     return K.mean(y_pred)

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

size=100
dummy = np.full(size,1)
# dummy2 = np.zeros(size,1)
# print(dummy)

model = load_model("SimNet.h5")
label = keras.utils.to_categorical(dummy, num_classes=2)


for i in read_root("CAT.root","tree",columns=["PMTALL"],flatten=True,chunksize=size):
	x = i.PMTALL
	x_np = x.values.reshape(size,4480)

	# x_tra = abs(np.transpose(x_np)-16000) #for FFT
	x_tra = abs(x_np-15200)
	extra = np.arange(4400,4480)
	x_trans = np.delete(x_tra,extra,axis=1)

	# wave = x.reshape(10,4480)
	# print(x_np)

	history = model.fit(x_trans,label,epochs=10,batch_size=64, verbose=1)
	scores = model.predict(x_trans,verbose=1)
	eva = model.evaluate(x_trans,label, verbose=1)

	# model.save("SimNet.h5")

	print(history.history.keys())
	# plt.plot(history.history["loss"])
	# plt.plot(history.history["acc"])
	# plt.show()

	# print(model.metrics_names, scores)

