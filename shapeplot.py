import numpy as np
import matplotlib.pyplot as plt

"""
MISSION IMPOSSIBLE of classification of gamma and beta, fix y1 to counting rather than just straight up shifting.
"""


x = np.arange(0,1,0.01)
y = np.exp(-x)

y1 = 0.1*np.exp(-(x+0.002*x))+0.2*np.exp(-(x+x*0.006))+0.7*np.exp(-x)

fig1 = plt.figure()
plt.bar(x,y,0.01,color="r",alpha=0.5)
plt.bar(x,y1,0.01,color="y",alpha=0.5)
plt.xlim(0.42,0.48)
plt.ylim(0.62,0.66)
plt.show()
fig2 = plt.figure()
plt.bar(x,y,0.01,color="r",alpha=0.2)
plt.bar(x,y1,0.01,color="y",alpha=0.2)
plt.show()

"""
add noise
"""

y_noise = (np.random.rand(len(x))*0.01-0.005)+y
y1_noise = (np.random.rand(len(x))*0.01-0.005)+y1

fig3 = plt.figure()
plt.bar(x,y_noise,0.01,color="r",alpha=0.9)
plt.bar(x,y1_noise,0.01,color="y",alpha=0.9)
plt.xlim(0.42,0.48)
plt.ylim(0.62,0.66)
plt.show()

# print(y_noise*0.01)

"""
Personal suggestion
create a set of false waveform pulse of beta, try to distinguish in from y and y1

add random generator to the ratio of gamma pulses
"""

Ratio = np.random.rand(3)
Ratio = Ratio/sum(Ratio)

Y = np.exp(-x) + (np.random.rand(len(x))*0.01-0.005)
Y1 = Ratio[0]*np.exp(-(x+np.random.rand(1)*0.002*x))+Ratio[1]*np.exp(-(x+np.random.rand(1)*x*0.006))+Ratio[2]*np.exp(-x) + (np.random.rand(len(x))*0.01-0.005)

fig4 = plt.figure()
plt.bar(x,Y,0.01,color="r",alpha=0.9)
plt.bar(x,Y1,0.01,color="y",alpha=0.9)
plt.xlim(0.42,0.48)
plt.ylim(0.62,0.66)
plt.show()


"""
Produce a set to randomly generated pulse shape based on the general pulse shape of beta and gamma
"""