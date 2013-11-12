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
#    phantom function : Copyright (C) 2010  Alex Opie  <lx_op@orcon.net.nz>
from pylab        import *
from scipy.linalg import toeplitz

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

def phantom (n = 256, p_type = 'Modified Shepp-Logan', ellipses = None):
  """
   phantom (n = 256, p_type = 'Modified Shepp-Logan', ellipses = None)
  
  Create a Shepp-Logan or modified Shepp-Logan phantom.

  A phantom is a known object (either real or purely mathematical) 
  that is used for testing image reconstruction algorithms.  The 
  Shepp-Logan phantom is a popular mathematical model of a cranial
  slice, made up of a set of ellipses.  This allows rigorous 
  testing of computed tomography (CT) algorithms as it can be 
  analytically transformed with the radon transform (see the 
  function `radon').
  
  Inputs
  ------
  n : The edge length of the square image to be produced.
  
  p_type : The type of phantom to produce. Either 
    "Modified Shepp-Logan" or "Shepp-Logan".  This is overridden
    if `ellipses' is also specified.
  
  ellipses : Custom set of ellipses to use.  These should be in 
    the form
      [[I, a, b, x0, y0, phi],
       [I, a, b, x0, y0, phi],
       ...]
    where each row defines an ellipse.
    I : Additive intensity of the ellipse.
    a : Length of the major axis.
    b : Length of the minor axis.
    x0 : Horizontal offset of the centre of the ellipse.
    y0 : Vertical offset of the centre of the ellipse.
    phi : Counterclockwise rotation of the ellipse in degrees,
          measured as the angle between the horizontal axis and 
          the ellipse major axis.
    The image bounding box in the algorithm is [-1, -1], [1, 1], 
    so the values of a, b, x0, y0 should all be specified with
    respect to this box.
  
  Output
  ------
  P : A phantom image.
  
  Usage example
  -------------
    import matplotlib.pyplot as pl
    P = phantom ()
    pl.imshow (P)
  
  References
  ----------
  Shepp, L. A.; Logan, B. F.; Reconstructing Interior Head Tissue 
  from X-Ray Transmissions, IEEE Transactions on Nuclear Science,
  Feb. 1974, p. 232.
  
  Toft, P.; "The Radon Transform - Theory and Implementation", 
  Ph.D. thesis, Department of Mathematical Modelling, Technical 
  University of Denmark, June 1996.
  
  """
  
  if (ellipses is None):
    ellipses = _select_phantom (p_type)
  elif (np.size (ellipses, 1) != 6):
    raise AssertionError ("Wrong number of columns in user phantom")
  
  # Blank image
  p = np.zeros ((n, n))

  # Create the pixel grid
  ygrid, xgrid = mgrid[-1:1:(1j*n), -1:1:(1j*n)]

  for ellip in ellipses:
    I   = ellip [0]
    a2  = ellip [1]**2
    b2  = ellip [2]**2
    x0  = ellip [3]
    y0  = ellip [4]
    phi = ellip [5] * pi / 180  # Rotation angle in radians
    
    # Create the offset x and y values for the grid
    x = xgrid - x0
    y = ygrid - y0
    
    cos_p = cos (phi) 
    sin_p = sin (phi)
    
    # Find the pixels within the ellipse
    locs = (((x * cos_p + y * sin_p)**2) / a2 
              + ((y * cos_p - x * sin_p)**2) / b2) <= 1
    
    # Add the ellipse intensity to those pixels
    p [locs] += I

  return p


def _select_phantom (name):
  if (name.lower () == 'shepp-logan'):
    e = _shepp_logan ()
  elif (name.lower () == 'modified shepp-logan'):
    e = _mod_shepp_logan ()
  else:
    raise ValueError ("Unknown phantom type: %s" % name)
  
  return e


def _shepp_logan ():
  #  Standard head phantom, taken from Shepp & Logan
  return [[   2,   .69,   .92,    0,      0,   0],
          [-.98, .6624, .8740,    0, -.0184,   0],
          [-.02, .1100, .3100,  .22,      0, -18],
          [-.02, .1600, .4100, -.22,      0,  18],
          [ .01, .2100, .2500,    0,    .35,   0],
          [ .01, .0460, .0460,    0,     .1,   0],
          [ .02, .0460, .0460,    0,    -.1,   0],
          [ .01, .0460, .0230, -.08,  -.605,   0],
          [ .01, .0230, .0230,    0,  -.606,   0],
          [ .01, .0230, .0460,  .06,  -.605,   0]]

def _mod_shepp_logan ():
  #  Modified version of Shepp & Logan's head phantom, 
  #  adjusted to improve contrast.  Taken from Toft.
  return [[   1,   .69,   .92,    0,      0,   0],
          [-.80, .6624, .8740,    0, -.0184,   0],
          [-.20, .1100, .3100,  .22,      0, -18],
          [-.20, .1600, .4100, -.22,      0,  18],
          [ .10, .2100, .2500,    0,    .35,   0],
          [ .10, .0460, .0460,    0,     .1,   0],
          [ .10, .0460, .0460,    0,    -.1,   0],
          [ .10, .0460, .0230, -.08,  -.605,   0],
          [ .10, .0230, .0230,    0,  -.606,   0],
          [ .10, .0230, .0460,  .06,  -.605,   0]]

#def ?? ():
# # Add any further phantoms of interest here
# return array (
#  [[ 0, 0, 0, 0, 0, 0],
#   [ 0, 0, 0, 0, 0, 0]])


