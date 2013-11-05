#
#    Copyright (C) <2013>  <cummings.evan@gmail.com>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
from pylab                            import *
from scipy.linalg                     import toeplitz

def wacky_thing(t, h, sig):
  """
  crazy function for x_true.
  """
  wack =  + 50 * (  0.75 * ((0.1 < t) & (t < 0.25)) \
                  + 0.25 * ((0.3 < t) & (t < 0.32)) \
                  + ((0.5 < t) & (t < 1)) * sin(2*pi*t)**4 )
  return wack / norm(wack)

def PSF(t, h, sig):
  """
  PSF reconstruction x_true.
  """
  sig1   = 0.02
  sig2   = 0.08
  # Create the left-half of the PSF
  mid    = int(round(len(t)/2))
  kernel = zeros(len(t))
  
  kernel[:mid] = exp(-(t[:mid])**2 / sig1**2)
  kernel[mid:] = exp(-(t[:mid])**2 / sig2**2)
  kernel[:mid] = kernel[:mid][::-1]
  
  # Create the normalized PSF
  return kernel / (h * sum(kernel))

def PSF2(t, h, sig):
  """
  PSF2 reconstruction x_true.
  """
  kernel  = zeros(len(t))
  low     = (-1/10. <= t) & (t <= 0)
  high    = (0      <  t) & (t <  1/10.)
  kernel[low]  =  100*t[low]  + 10
  kernel[high] = -100*t[high] + 10
  return kernel

def gaussian_PSF(t, h, sig):
  """
  Gaussian kernel PSF.
  """
  kernel = 1/(sqrt(pi)*sig) * exp(-t**2/(sig**2))
  return kernel

def gaussian_PSF_2D(x, y, hx, hy, sig):
  """
  Gaussian kernel PSF.
  """
  kernel = exp(-((x)**2 + (y)**2) / (2*sig**2))
  kernel = kernel / sum(kernel)
  return kernel

def descritize_PSF_kernel(t, h, kernel):
  """
  Descritization of the Gaussian.
  """
  A = h * toeplitz(kernel)
  return A 

def descritize_integral(t, h):
  """
  Descritization of the integral operator.
  """
  n = len(t)
  A = h * tril(ones((n,n)))
  return A



