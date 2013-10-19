from pylab          import *
from scipy.linalg   import toeplitz

def wacky_thing(t):
  """
  crazy function for x_true.
  """
  wack =  + 50 * (  0.75 * ((0.1 < t) & (t < 0.25)) \
                  + 0.25 * ((0.3 < t) & (t < 0.32)) \
                  + ((0.5 < t) & (t < 1)) * sin(2*pi*t)**4 )
  return wack / norm(wack)

def PSF(t, sig=[0.02, 0.08]):
  """
  PSF reconstruction x_true.
  """
  h      = (max(t) - min(t)) / len(t) 
  sig1   = sig[0]
  sig2   = sig[1]
  # Create the left-half of the PSF
  mid    = int(round(len(t)/2))
  kernel = zeros(len(t))
  
  kernel[:mid] = exp(-(t[:mid])**2 / sig1**2)
  kernel[mid:] = exp(-(t[:mid])**2 / sig2**2)
  kernel[:mid] = kernel[:mid][::-1]
  
  # Create the normalized PSF
  return kernel / (h * sum(kernel))

def PSF2(t, sig=None):
  """
  PSF2 reconstruction x_true.
  """
  kernel  = zeros(len(t))
  low     = (-1/10. <= t) & (t <= 0)
  high    = (0      <  t) & (t <  1/10.)
  kernel[low]  =  100*t[low]  + 10
  kernel[high] = -100*t[high] + 10
  return kernel

def gaussian_PSF(t, sig=0.05):
  """
  Gaussian kernel PSF.
  """
  n      = float(len(t))
  h      = (max(t) - min(t)) / n 
  kernel = 1/(sqrt(pi)*sig) * exp(-t**2/(sig**2))
  return kernel

def descritize_PSF_kernel(t, kernel):
  """
  Descritization of the Gaussian.
  """
  n = len(t)
  h = (max(t) - min(t)) / n 
  A = h * toeplitz(kernel)
  return A 

def descritize_integral(t):
  """
  Descritization of the integral operator.
  """
  n = len(t)
  h = (max(t) - min(t)) / n 
  A = h * tril(ones((n,n)))
  return A



