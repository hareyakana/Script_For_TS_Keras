import matplotlib.pyplot as plt
from pandas import Series
import pandas as pd
from root_pandas import read_root
import numpy as np
from sklearn.preprocessing import normalize

pmtall = read_root("Alpha_11.root", "tree", columns=["PMTALL"],flatten=True)
energy = read_root("Alpha_11.root", "tree", columns=["Energy"],flatten=True)

"""
functions for root pandas
"""

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

def root_single(var,pos):
	"""Get the correct size for variables[1] etc."""
	array = []
	output = []
	variables = var.values
	for x,y in enumerate(variables):
		if y[1] == pos:
			output=(np.append(output,y[0]))
	# output_abs = abs(output-15200)
	# output_nor = [float(i)**2/sum(output_abs) for i in output_abs]
	return output

"""
Time series per event"
"""
# import statsmodels.api as sm

# from pandas.plotting import lag_plot

TX = NORMPMTALL(pmtall,1572)

# from datetime import datetime
# import markovify
from discreteMarkovChain import markovChain

for i in range(10):
	diff = Series(np.diff(TX[i,0:4400]))
	full = Series(TX[i,0:4400])
	# print(full.shape)
	# dam = sm.tsa.datetools.dates_from_range('1700', length=len(TX[i,0:4400]))
	# dates = sm.tsa.datetools.dates_from_range('2000', length=len(full))
	# dates = np.arange(0,4400)

	# dual = np.zeros((2,4400))
	# for n in range(4400):
	# 	dual[0][n] = dates[n]

	# for n in range(4400):
	# 	dual[1][n] = full[n]

	# quantiles = np.arange(10) #q1,q2,q3. ...q10, easier to work with integers in python
	# quantiles = np.array(["q1","q2","q3","q4","q5","q6","q7","q8","q9","q10"])
	# print(quantiles)

	bins = np.arange(0,0.101,0.001)
	# print(bins)

	# dual = []
	# for k in range(len(full)):
	# 	z = full[k]
	# 	for p in range(10):
	# 		if z>=bins[p] and z<=bins[p+1]:
	# 			dual.append(quantiles[p])

	# dual = np.array(dual)
	# print(dual)

	inds = np.digitize(full,bins)
	# print(inds)

	# lags = np.zeros((4400,2))
	lags=[]
	for x in range(len(inds)-1):
		lags.append([inds[x],inds[x+1]])
		# lags[x][0] = "{},".format(inds[x])
		# lags[x][1] = inds[x+1]
	lags = np.array(lags)
	# lags = np.transpose(lags)
	# print(lags)
	# print(lags.shape)

	# mc = markovChain(lags[0],lags[1])
	# mc.computePi('eigen') #We can also use 'power', 'krylov' or 'eigen'
	# print(mc.pi)


	counts, bin_edge = np.histogram(inds)
	# print(counts)

	dual = np.zeros((len(counts),len(counts)))
	for x in range(len(dual)):
		for y in range(len(dual[x])):
			dual[x][y] = counts[x] + counts[y]

	# for x in range(len(dual)):
	# 	print(dual[x])
	dual = normalize(dual,norm="l2")

	# print(dual)
	plt.imshow(dual)
	plt.show()

	# print(counts)

	# plt.plot(inds)
	# plt.show()


	# q = full.hist()
	# print(q.hist())


	# for n in range(4400):
	# 	dual[2][n] = full[n]

	# print(dates.shape)
	# print(dual)
	# dual = np.transpose(dual)

	# full = Series(TX[i,0:4400],index=dates)
	# print(full)
	# print(full)
	# del full["YEAR"]

	# comb = Series(TX[i,0:4400],index=dam)
	# print(dual[1][1].shape)
	# damb = sm.tsa.ARMA(dual,order=(1,0,1))
	# aram = damb.fit(disp=0)
	# print(aram.summary())

	# test_model = markovify.Text(np.array_str(TX[i]))
	# print(test_model)







