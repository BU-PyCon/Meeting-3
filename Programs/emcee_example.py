"""
This code provides a start to finish example of how to use emcee - the MCMC Hammer

For information on emcee - visit http://dan.iel.fm/emcee/current/
You may notice that this example is similar to the one on this site...
"""

#Start by importing some useful modules
import numpy as np
import emcee
import matplotlib.pyplot as plt
import sys   #This is only needed to do some fancy print to track the MCMC progress
import triangle

#In this example, we're going to fit a model to some data resembling a sine curve.
#Let's say the sine curve will be of the form y = A*sin(x-B), where A and B are parameters.
#To start, let's make some fake, noisy (and arbitrary) data.
A_true, B_true = 4.3, 1.5
x = np.linspace(0, 4*np.pi, 100)
data = A_true*np.sin(x - B_true)

#Now add some random Gaussian noise to the data. Let's say each data point has an error
#  value called 'err'. For the sake of this example, let's say err=2.
err = np.ones([np.size(x)])*2. 
noise = np.random.normal(loc=0, scale=2, size=np.size(x))
data_noisy = data + noise

#Now plot both the data and noisy data for a sanity check
plt.figure(1)
plt.plot(x, data, color='r', linewidth=2)
plt.errorbar(x, data_noisy, yerr=err, ecolor='k',fmt='o', color='k')
plt.show()

# stop = input('Hit return to continue\n')

#MOVING ON...
#We have our data with errorbars, we have our model (a sine curve with 2 free parameters),
#  so now we're ready to start using emcee!
#Remember how I mentioned a likelihood function and a prior distribution? You need to make
#  a separate function for each of these. And each are the natural log.

#Start with the likelihood function. This is the probability that the data resulted from
#  the model, which is a function of your parameters. We're going to assume the likelihood
#  is simply a Gaussian with mean and standard deviation set by the model and the error on 
#  the data. This is usually an OK assumption (cf. Central Limit Theorem), but it may not
#  always be appropriate. 
def ln_likelihood(parameters, x, data_noisy, err):
	#Unpack your parameters and make your model
	A, B = parameters
	model = A*np.sin(x - B)
	
	#Now we can return the ln of the likelihood function (which is a Gaussian). 
	#The sigma in a normal Gaussian is just the error on our data points and the mu is 
	#  just our model. (It may require doing out the algebra to understand the next line)
	return -0.5*np.sum(((data_noisy-model)/err)**2 + np.log(2*np.pi*err**2))

#The next function to define is the priors. You can decide if you want you priors to be 
#  'uninformative' - just a range in parameter space - or to have some other distribution.
#  Remember, a prior can penalize the likelihood function. That means a very low 
#  probability or, in our ln case, a very large negative number. For emcee, this is -inf
def ln_prior(parameters):
	#So what I want to say is A must be in in the range (0,10) and B must be in the range 
	#  (0,pi). If not, return -inf and that set of parameters gives a model with 
	#  probability of zero. If so, I return zero. Remember that ln(1)=0. You could have 
	#  more informed priors that return zero for parameters you want to weigh highly or 
	#  slightly negative numbers, say ln(0.5)=-0.7, for parameters that are less likely
	#  but still not terrible. 
	A, B = parameters
	if 0 < A < 10 and 0 < B < np.pi:
		return 0.
	else:
		return -np.inf
		
#Now you're ready to find the full ln probability function, which is just the sum of the 
#  two previous functions with a check to make sure we don't have a -inf
def ln_probability(parameters, x, data_noisy, err):
    #Let's also keep track of the iterations, just for fun
    global iteration
    iteration +=1
    sys.stdout.write("Sampler progress: %d%%   \r" % \
    	(100.*iteration/(n_walkers*n_steps + n_walkers -1)))
    sys.stdout.flush()
    
    #Call the prior function, check that it doesn't return -inf then create the log-
    #  probability function
    priors = ln_prior(parameters)
    if not np.isfinite(priors):
        return -np.inf
    else:
    	return priors + ln_likelihood(parameters, x, data_noisy, err)

#Almost there! Now we must initialize our walkers. Remember that emcee uses a bunch of 
#  walkers, and we define their starting distribution. If you have an idea of where your
#  best-fit parameters will be, you can start the walkers in a small Gaussian bundle 
#  around that value (as I am doing). Otherwise, you can start them evenly across your
#  parameter space (that is limited by the priors). This will require more walkers and 
#  more steps.
n_dim, n_walkers, n_steps, iteration = 2, 250, 300, -1
A_guess, B_guess = 6., np.pi/4.

A_rand = np.random.normal(loc=A_guess, scale=0.2*A_guess, size=n_walkers)
B_rand = np.random.normal(loc=B_guess, scale=0.2*B_guess, size=n_walkers)

#positions should be a list of N-dimensional arrays where N is the number of parameters 
#  you have. The length of the list should match n_walkers.
positions = []
for param in range(n_walkers):
	positions.append(np.array([A_rand[param],B_rand[param]]))

#Finally, you're ready to set up and run the emcee sampler
sampler = emcee.EnsembleSampler(n_walkers, n_dim, ln_probability, args=(x, data_noisy, err))
sampler.run_mcmc(positions,n_steps)


#Callooh! Callay! - our MCMC has finished. Now how do we extract the results?
#All of the walker information is saved in sampler.chain. Do a np.shape() to see. 
#Let's plot up the path of each walker in terms of each variable. I choose to do this in
#  a loop - there is a likely a better way. Get creative! 
chain_shape = np.shape(sampler.chain) 

figure2, axes = plt.subplots(2, sharex=True, figsize=(8,6))

#Y-axis labels and other pleasantries
axes[0].set_title('Walker Paths', fontsize='large')
axes[0].set_ylabel('$A$',fontsize='large')
axes[1].set_ylabel('$B$',fontsize='large')
axes[1].set_xlabel('Step Number', fontsize='large')
axes[1].set_xlim(0,chain_shape[1])

#Plot the walkers
for walkers in range(chain_shape[0]):
	for params in range(chain_shape[2]):
		axes[params].plot(sampler.chain[walkers,:,params], linewidth=0.5, alpha=0.5,\
			color='k')
plt.show()

#You may notice that the walkers took a few steps to find the local probability well.
#  Before we calculate numbers and uncertainties, we want to remove that burn in.
burn = float(input('Enter the burn_in length: '))
#Now apply the burn by cutting those steps out of the chain. 
chain_burnt = sampler.chain[:, burn:, :] 
#Also flatten the chain to just a list of samples
samples = chain_burnt.reshape((-1, n_dim))


#There's another very interesting plot you can make: a triangle plot. For this you will
#  need to install triangle_plot. This type of plot is very frequently used to display 
#  posterior probability distributions of all the parameters. It very quickly shows any
#  covariances between parameters and allows for you to see the 1D posterior distributions
#  for a single parameter, marginalized over all the others.
figure3 = triangle.corner(samples, labels=["$A$", "$B$"],truths=[A_true, B_true])
plt.show()

# stop = input('Hit return to continue\n')

#So that's all well and good, but what are my best-fit parameter values and uncertainties?

#This rather 'pythonic' line came straight from the emcee documentation. It sets up a 
#  function with lambda and applies that functions to each percentile to get the best-
#  fit values.
A_mcmc, B_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samples,\
	[16, 50, 84],axis=0)))
	
print('')
print(A_mcmc, B_mcmc)

#And you're done! The above variables print out the parameter values and asymmetric errors.
	















 
