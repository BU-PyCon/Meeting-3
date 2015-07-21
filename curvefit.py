import numpy as np
import matplotlib.pyplot as plt
from random import random
from scipy.optimize import curve_fit

def gaussian(x, m, s):
    """
    A normalized gaussian function with
    mean, m, and standard deviation, s.
    """
    return np.exp(-0.5*((x-m)/s)**2) / (s * np.sqrt(2*np.pi))

#Let's generate a true gaussian curve
mean = 5
stddev = 3
x_true = np.arange(-10,20,0.1)
y_true = gaussian(x_true, mean, stddev)

#Now let's generate "data" by perturbing the true values
x_data = np.arange(-10,20,0.5)
y_data = gaussian(x_data, mean, stddev)
y_data += [(random()*2-1)*0.05 for i in range(len(y_data))]

#Now let's fit a curve to our data and see how well it compares
x_fit = np.arange(-10,20,0.1)
popt, pcov = curve_fit(gaussian, x_data, y_data)
y_fit = gaussian(x_fit, popt[0], popt[1])
#I could have called curve_fit with p0 as a parameter which would be an array
#of values equal to my initial guess for the optimal parameters. Note, if
#your function is multi-variate, then x_data should be a M x N array where N
#is the number of data points and M is the number of variables. Currently
#M = 1 so x_data is a 1 x N array.


plt.figure(1)
plt.plot(x_true, y_true, '-', lw = 3, c = (0.694,0.906,0.561),
         label = 'Actual: $\mu$ = '+str(mean)+', $\sigma$ = '+str(stddev))
plt.plot(x_data, y_data, ' ob', label = '"Data"')
plt.plot(x_fit, y_fit, '-r',
         label = 'Fit: $\mu$ = '+str(round(popt[0],3))+
                 ', $\sigma$ = '+str(round(popt[1],3)))
plt.legend(loc = 0)
plt.show(block = False)
