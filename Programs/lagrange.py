import numpy as np
import matplotlib.pyplot as plt
from random import random
from scipy.interpolate import lagrange

def func(x):
    return 5*x*x - 3*x + 2

#Define the true curve
x_true = np.arange(-2,2.1,0.1)
y_true = func(x_true)

#Define 5 "data points" from this curve, with noise
x_data = np.arange(-2,3)     #Define a set of five x points
y_data = func(x_data)        #Define the y points from an explicit function
y_data += [(random()*2-1)*2 for i in range(len(y_data))]  #Add some random noise

#Define the lagrangian interpolated curve
x_interp = np.arange(-2,2.1,0.1)
poly = lagrange(x_data, y_data) #Lagrange returns a function that can be called
y_interp = poly(x_interp)

#Now plot everything up
plt.figure(1)
plt.plot(x_true,y_true,'-',lw=2,label = 'True Curve')
plt.plot(x_data,y_data,' o', label = '"Measured Data"')
plt.plot(x_interp,y_interp,'-k', label = 'Lagrange Interpolation')
plt.xlim(-2.5,2.5)
plt.legend(loc=0)
plt.show(block = False)
