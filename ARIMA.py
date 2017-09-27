import matplotlib.pyplot as plt
from pandas import Series
import pandas as pd
from root_pandas import read_root
import numpy as np
from sklearn.preprocessing import normalize

pmtall = read_root("RefPulse.root", "tree", columns=["PMTALL"],flatten=True)
energy = read_root("RefPulse.root", "tree", columns=["Energy"],flatten=True)

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
import statsmodels.api as sm

# from pandas.plotting import lag_plot

TX = NORMPMTALL(pmtall,1996)

# from datetime import datetime
# import markovify
from discreteMarkovChain import markovChain

for i in range(10):
	diff = np.diff(TX[i,:])
	full = TX[i,:]

	# when comparing AIC/BIC/HQIC, the relative magnitude and sign does not matter, what matters is that the lower value is generally perferred for the ARIMA model

	damb = sm.tsa.ARIMA(diff, order=(1,1,1))
	aram = damb.fit(disp=0)
	print(aram.summary())
	damb = sm.tsa.ARIMA(diff, order=(2,1,1))
	aram = damb.fit(disp=0)
	print(aram.summary())
	damb = sm.tsa.ARIMA(diff, order=(3,1,1))
	aram = damb.fit(disp=0)
	print(aram.summary())
	damb = sm.tsa.ARIMA(diff, order=(2,1,2))
	aram = damb.fit(disp=0)
	print(aram.summary())
	damb = sm.tsa.ARIMA(diff, order=(2,1,3))
	aram = damb.fit(disp=0)
	print(aram.summary())
	damb = sm.tsa.ARIMA(full, order=(1,1,1))
	aram = damb.fit(disp=0)
	print(aram.summary())
	damb = sm.tsa.ARIMA(full, order=(1,2,1))
	aram = damb.fit(disp=0)
	print(aram.summary())
	damb = sm.tsa.ARIMA(full, order=(2,1,1))
	aram = damb.fit(disp=0)
	print(aram.summary())
	damb = sm.tsa.ARIMA(full, order=(1,1,2))
	aram = damb.fit(disp=0)
	print(aram.summary())




	break
		# print(test_model)







