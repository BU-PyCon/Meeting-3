import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.interpolate import interp1d,splev,splrep

def extractSpectrum(filename):
    """
     NAME:
       extractSpectrum

     PURPOSE:
       To open an input fits file from SDSS and extract the relevant
       components, namely the flux and corresponding wavelength.

     INPUTS:
       filename    The path and filename (including the extension) to the
                   file to be read in.

     OUTPUTS:
       lam         The wavelengths, in angstrom, of the flux values
       flux        The actual flux, in arbitrary units

     EXAMPLE:
       flux, lam = extractSpectra('path/to/file/filename.fits')
    """
    hdu = fits.open(filename)       #Open the file using astropy
    data = hdu[1].data              #Data is in 2nd component of HDU
    flux = data['flux']             #Get flux from read in dict
    lam = 10**(data['loglam'])      #Get wavelength, make it not log10
    hdu.close()                     #Close the file, we're done with it
    return lam, flux                #Return the values as numpy arrays

def interpolate(points, lam, flux, method):
    """
     NAME:
       interpolate

     PURPOSE:
       General purpose function that can call and use various scipy.interpolate
       methods. Defined for convienience.

     INPUTS:
       points      Set of new points to get interpolated values for.
       lam         The wavelengths of the data points
       flux        The fluxes of the data points
       method      The method of interpolation to use. Valide values include
                   'interp1d:linear', 'interp1d:quadratic', and 'splrep'.

     OUTPUTS:
       Interpolated set of values for each corresponding input point.

     EXAMPLE:
       interpFlux = interpolate(interpLam, lam, flux)
    """
    if method == 'interp1d:linear':
        f = interp1d(lam, flux, assume_sorted = True)
        return f(points)
    if method == 'interp1d:quadratic':
        f = interp1d(lam, flux, kind = 'quadratic', assume_sorted = True)
        return f(points)
    if method == 'splrep':
        return splev(points, splrep(lam, flux))
    raise Exception("You didn't choose a proper interpolating method")

#First extract the flux and corresponding wavelength
fileName = 'spec-4053-55591-0938.fits'
lam, flux = extractSpectrum(fileName)

#Now let's plot it, without any processing
plt.figure(1)
plt.plot(lam, flux, '-o', lw = 1.5, c = (0.694,0.906,0.561),
         mec = 'none', ms = 4, label = 'Original data')
plt.xlabel('Wavelength', fontsize = 16)
plt.ylabel('Flux', fontsize = 16)
plt.ylim(0,1.1*max(flux))

#Now let's interpolate and plot that up
interpLam = np.arange(4000,10000,1)
interpFlux = interpolate(interpLam, lam, flux, 'splrep')  #This is my own method
plt.plot(interpLam, interpFlux, '-k', label = 'Interpolated')

plt.legend(loc = 0)
plt.show(block = False)

print('Done...')
