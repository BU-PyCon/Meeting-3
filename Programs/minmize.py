import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
from time import time
import warnings
warnings.filterwarnings("ignore")


#This is the function we want to minimize
def func(x):
    """
    The Rosenbrock function
    """
    return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2

#These are the jacobian and hessian of the function which are
#necessary for some minimization techniques.
def func_jac(x):
    """
    The Jacobian of the Rosenbrock function,
    in this case, just the gradient
    """
    jac = np.zeros([2])
    jac[0] = -2*(1-x[0]) - 400*(x[1]-x[0]**2)*x[0]
    jac[1] = 200*(x[1]-x[0]**2)
    return jac

def func_hess(x):
    """
    The Hessian of the Rosenbrock function
    """
    hess = np.zeros([2,2])
    hess[0,0] = 2 - 400*(x[1]-x[0]**2) + 800*x[0]**2
    hess[0,1] = -400*x[0]
    hess[1,0] = -400*x[0]
    hess[1,1] = 200
    return hess
    

#Define our data points for the plot
x = np.arange(-2.5,2.6,0.1)
y = np.arange(-1,4.1,0.1)
x, y = np.meshgrid(x,y)     #This is a useful function, you should look it up!
z = func([x,y])

#Let's plot it up to see what it looks like
fig = plt.figure(1)
ax = fig.gca(projection='3d')
ax.plot_surface(x,y,z,rstride=1,cstride=1,cmap=cm.Paired,linewidth = 0,antialiased=False)
ax.view_init(elev=32,azim=130)
plt.show(block = False)

#Now let's find the minimum
startPoint = [2,-4] #Set the starting (x,y) point
print('Finding minimum of function...')
print('Actual minimum is at:\n\tx = 1.0000\n\ty = 1.0000\nwhere f(1,1) = 0.0000\n')

#Nelder-Mead (Downhill Simplex) Method
"""
The Nelder-Mead method, otherwise known as the Downhill Simplex method, uses
a simplex to "crawl" around the parameter space and over many iterations, it
tends to "fall" towards the minimum according to a set number of rules about
how it "crawls". In the end, it will bound the minimum. The major usage of
this method is that it does not require a derivative as it is not a gradient
descent method.
"""
print('=== Nelder-Mead (Downhill Simplex) Method ===')
start = time()
res = minimize(func, startPoint, method='nelder-mead',
               options = {'xtol':1E-8,'disp':True})
print('Process ran in',time()-start,'seconds.')
print('Minimum found at:\n\tx =',res.x[0],'\n\ty =',res.x[1],'\n')

#Powell (Direction Set) Method
"""
The Powell method, otherwise known as the direction set method, essentially
minimizes a function of multiple variables one direction at a time. In effect,
it will hold all variables of the function constant, except for one of them,
and minimize that single variable function. Once that minimum is found, it
moves on to the next variable, holding all the rest fixed, and minimizes that.
This is repeated until no amount of minimizing moves the point more than
the tolerance. This method suffers from the fact that the set of directions
it may choose to use will not be optimal and so it may not choose the most
efficient route down a "valley" in the function. Again though, this does not
require knowing the gradient and so can still be useful.
"""
print('=== Using Powell Method ===')
start = time()
res = minimize(func, startPoint, method='powell',
               options = {'xtol':1E-8,'disp':True})
print('Process ran in',time()-start,'seconds.')
print('Minimum found at:\n\tx =',res.x[0],'\n\ty =',res.x[1],'\n')

#Conjugate Gradient Method
"""
This method uses something related to the gradient descent method, but tries
to be a bit smarter about it. Effectively, if we know the gradient of the
function near our current point, we can determine which direction is "downhill".
This can be used to judge the best direction to travel. However, this may run
into the same problem as the Powell method in that it does not properly move
down long thin valleys in the function. This is where the conjugate component
comes into play. Rather than moving exactly along the gradient, it chooses
a path that is determined by a variety of factors including the gradient and
the last path directions travelled. The gradient in this method is passed in
as a jacobian.
"""
print('=== Using Conjugate Gradient Method ===')
start = time()
res = minimize(func, startPoint, method = 'cg', jac = func_jac,
               options = {'disp':True})
print('Process ran in',time()-start,'seconds.')
print('Minimum found at:\n\tx =',res.x[0],'\n\ty =',res.x[1],'\n')

#Broyden-Fletcher-Goldfarb-Shanno (BFGS) Method
"""
This is a type of Quasi-Newton method and has a similar goal as the conjugate
gradient method and as such also requires knowing the gradient. The main
difference here is that more information is used in choosing a new descent
direction. 
"""
print('=== Using BFGS Method ===')
start = time()
res = minimize(func, startPoint, method = 'bfgs', jac = func_jac,
               options = {'disp':True})
print('Process ran in',time()-start,'seconds.')
print('Minimum found at:\n\tx =',res.x[0],'\n\ty =',res.x[1],'\n')


#Newton Conjugate Gradient Method
"""
Similar to the Conjugate Gradient method, except that it also uses the second
derivative of the function (for multivariate functions, the hessian) as well
as the gradient (jacobian for multivariate). With this extra information,
it is able to more aptly choose the direction of travel when searching for
minimums because it can better model the function (now we can taylor expand to
include 3 terms). This has the added advantage that you don't need as many
function calls, but the detraction that you have to know an explicit function
for both the first and second derivatives of your function, which may not
be calculable.
"""
print('=== Using Newton Conjugate Gradient Method ===')
start = time()
res = minimize(func, startPoint, method = 'newton-cg',
               jac = func_jac, hess = func_hess,
               options = {'disp':True})
print('Process ran in',time()-start,'seconds.')
print('Minimum found at:\n\tx =',res.x[0],'\n\ty =',res.x[1],'\n')

#Sequential Least SQuares Programming Method
"""
This is a method which is capable of performing minimization according to
constraint funtions. Behind the scences, it uses both the Powell and BFGS
methods. In this case, we don't need the jacobian though. The constraints
are passed in as a dict, telling the type of function (the constraint
function should always equal zero). In this case, we minimize the same
Rosenbrock function, but subject to the constraint that the minimum lie
on the unit sphere.
"""
#Set up the constraint
cons = ({'type':'eq', 'fun':lambda x: x[0]*x[0] + x[1]*x[1] - 1})
print('=== Using SLSQP Method ===')
print('Minimum has been additionally constrained to lie on unit sphere.')
res = minimize(func, startPoint, method = 'slsqp',
               constraints = cons, options = {'disp':True})
print('Minimum found at:\n\tx =',res.x[0],'\n\ty =',res.x[1])
print('Minimum meets constraint:\n\tx^2 + y^2 =',res.x[0]**2 + res.x[1]**2)
