import matplotlib.pyplot as plt
from pandas import Series
import pandas as pd
from root_pandas import read_root
import numpy as np
from sklearn.preprocessing import normalize
import statsmodels.api as sm

pmtall = read_root("CAT.root", "tree", columns=["PMTALL"],flatten=True)
energy = read_root("CAT.root", "tree", columns=["Energy"],flatten=True)

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
from pandas.plotting import lag_plot, autocorrelation_plot

energy1 = root_single(energy, 1)

TX = NORMPMTALL(pmtall,41260)

for i in range(10):
	diff = Series(np.diff(TX[i,0:4400]))
	full = Series(TX[i,0:4400])
	# log = Series(np.log(TX[0]))
	"""lag plot stationary"""
	fig1 = plt.figure()
	lag_plot(diff, s=1, c="k")
	plt.title("lag_plot stationarized waveform, Energy={}keV".format(energy1[i]))
	plt.savefig("{0:03}_diff_lag.png".format(i))
	plt.close(fig1)

	"""lag plot waveform"""
	fig2 = plt.figure()
	lag_plot(full, s=1, c="k")
	plt.title("lag_plot, Energy={}keV".format(energy1[i]))
	plt.savefig("{0:03}_full_lag.png".format(i))
	plt.close(fig2)

	"""Simple stationary waveform"""
	fig3 = plt.figure()
	diff.plot()
	plt.title("stationarized waveform, Energy={}keV".format(energy1[i]))
	plt.savefig("{0:03}_diff_waveform.png".format(i))
	plt.close(fig3)

	"""Simple waveform"""
	fig4 = plt.figure()
	full.plot()
	plt.title("Waveform, Energy={}keV".format(energy1[i]))
	plt.savefig("{0:03}_full_waveform".format(i))
	plt.close(fig4)

	"""Partial Autocorrelation stationarized"""
	fig5 = plt.figure()
	# autocorrelation_plot(diff)
	sm.graphics.tsa.plot_pacf(diff, lags=30)
	plt.title("Autocorrelation stationarized waveform, Energy={}keV".format(energy1[i]))
	plt.savefig("{0:03}_diff_autocor".format(i))
	plt.close(fig5)

	# """Auto Correlation """
	# fig6 = plt.figure()
	# autocorrelation_plot(full)
	# sm.graphics.tsa.plot_pacf(full, lags=30)
	# plt.title("Autocorrelation, Energy={}".format(energy1[i]))
	# plt.savefig("{0:03}_full_autocor.png".format(i))
	# plt.close(fig6)

	""" Moving Average """
	fig7 = plt.figure()
	ma = full.rolling(10).mean()
	mstd = full.rolling(10).std()
	plt.plot(full.index,full)
	plt.plot(ma.index,ma)
	plt.fill_between(mstd.index, ma-2*mstd, ma+2*mstd, alpha=0.2)
	plt.savefig("{0:03}_full_ma.png".format(i))
	plt.close(fig7)


	from datetime import datetime

	# dam = sm.tsa.datetools.dates_from_range('1700', length=len(TX[i,0:4400]))
	full.index = pd.Index(sm.tsa.datetools.dates_from_range('1700', length=len(full)))
	# del full["YEAR"]

	# comb = Series(TX[i,0:4400],index=dam)
	
	aram = sm.tsa.ARMA(full,(2,0),freq ="B").fit(disp=False)
	print(aram)