import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from root_pandas import read_root
pmtall = read_root("RefPulse.root","tree",columns=["PMTALL"],flatten=["PMTALL"])

from sklearn.preprocessing import normalize

def NORMPMTALL(pmtall,num_entries):
	"""
	Convert read_root into numpy for speed
	invert the PMT pulse
	get rid of the last 80 bin as those bins of the pulse do not matter
	"""
	x = pmtall.PMTALL.values.reshape(num_entries,4480)
	x_abs = abs(x-15200)
	extra = np.arange(4400,4480)
	x_mod = np.delete(x_abs,extra,axis=1)
	x_nor = normalize(x_mod,norm="l2")
	# print(np.sum(x_nor,axis=1))
	return x_nor

PMTALL = NORMPMTALL(pmtall,1996)

from pandas import Series
from scipy import linalg

# from discreteMarkovChain import markovChain as mC

# REDO!! 

# Something not right, should be symmetrical around the diagonal.



# using basinhopping in scipy to solve multidimension equation
# then normalized results per row


from pandas.plotting import lag_plot
from scipy import stats

def markov(pmtall):
	x = pmtall
	w,h = x.shape
	# print(w,h)
	# print(x[1])
	# print(x[2])

	for i in range(len(x)):

		temp = x[i]
		# print(temp)
		# print(x[1])
		# print(t.shape)

		""" method1 """

		# bins = np.arange(0, 0.101, 0.001)
		# inds = np.digitize(tempe, bins)
		# print(inds)
		# print(len(inds))
		# counts, bin_edge = np.histogram(inds,bins=len(bins))
		# print(counts)

		# dual = np.zeros((len(counts),len(counts)))

		# for x in range(len(dual)):
		# 	for y in range(len(dual[x])):
		# 		dual[x][y] = counts[x] + counts[y]

		# dual = normalize(dual,norm="l1")

		# plt.imshow(dual, interpolation="nearest")
		# plt.show()

		# w = np.zeros((h,h))
		# for p in range(h):
		# 	for q in range(h):
		# 		t1 = t[p]
		# 		t2 = t[q]
		# 		w[p][q] = (t1 + t2)*abs(p - q)/h

		# z = Series((t,t.shift(1)))
		# print(z)
		# print(z.shape)
		# print(w)
		# plt.imshow(w)
		# plt.show()
		# if (i>5):
			# break



		"""
		method2
		construct a Markov Matrix/ Stochastic Matrix
		"""
		# produce a lag series, then count the probability of each quantile of the next quantile
		bins = np.arange(0, 0.080, 0.001)
		inds = np.digitize(temp, bins, right=True)
		print(inds)
		print(len(inds))
		print(max(inds))

		lags = np.zeros((2,len(inds)-1))
		for y in range(len(inds)-1):
			lags[0][y] = inds[y]
			lags[1][y] = inds[y+1]
		print(lags)
		print(lags.shape)

		# # ####
		# unique_elements, counts_elements = np.unique(inds, return_counts=True)
		# # print(counts_elements)

		# counting = np.zeros((2,len(bins)))
		# for y in range(len(bins)):
		# 	if y >= len(unique_elements):
		# 		break
		# 	counting[0][y] = unique_elements[y]
		# 	counting[1][y] = counts_elements[y]

		# print(counting)
		# # ######
			



		MTM = np.zeros((len(bins),len(bins)))
		for y in range(len(bins)):
			for z in range(len(bins)):

				dummy = 0
				for p in range(len(lags[0])):
					if lags[0][p] == y and lags[1][p] == z :
					# if (lags[0][p] == y or lags[0][p] == z) and (lags[1][p] ==y or lags[1][p] == z):
						# print(y)
						# print(lags[0][p])
						dummy += 1
				# print(dummy)

				MTM[y][z] = dummy
				# unique_elements, counts_elements = np.unique(dummy, return_counts=True)

				# print(unique_elements, counts_elements)
				# break
		# 		MTM[y][z] == 1



				 #figure how to calculate the probability
		MTM = normalize(MTM,norm="l1")
		print(MTM)

		plt.imshow(MTM,interpolation="nearest")
		plt.title("Markov Transition Matrix")
		plt.show()

		break

	return 1

markov(PMTALL)