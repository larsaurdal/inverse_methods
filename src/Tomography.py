import sys
src_directory = '../src/'
sys.path.append(src_directory)

from pylab                   import *
from functions               import phantom
from scipy.sparse            import csr_matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable

class Tomography(object):

  def __init__(self, n, ntheta, nz, err_lvl, cmap='gray'):
    """
    initializes and prepares the system for solving.
    """
    x_true    = phantom(n, p_type = 'Modified Shepp-Logan')
    theta     = linspace(-pi/2, pi/2, ntheta) 
    z         = linspace(-0.50, 0.50, nz) 
    Z, Theta  = meshgrid(z,theta)
    
    # stack the columns to make a single vector : 
    vecZ      = reshape(Z,      ntheta*nz, 'F')
    vecTheta  = reshape(Theta,  ntheta*nz, 'F')
    vecX      = reshape(x_true, n*n,       'F')
   
    # perform radon transformation :
    A         = self.Xraymat(vecZ, vecTheta, n) 
    Ax        = A.dot(vecX)
    noise     = err_lvl/100 * norm(Ax) / sqrt(ntheta*nz)
    b         = reshape(Ax,(nz,ntheta),'F') + noise*randn(nz,ntheta)
    vecb      = reshape(b, ntheta*nz, 'F')

    self.n        = n
    self.ntheta   = ntheta
    self.nz       = nz
    self.x_true   = x_true
    self.b        = b
    self.Ax       = Ax
    self.A        = A
    self.vecb     = vecb
    self.vecZ     = vecZ
    self.vecTheta = vecTheta
    self.vecX     = vecX
    self.cmap     = cmap
   
  def plot_true(self, ax):
    """
    plot the true and blurred solution.
    """
    x_true  = self.x_true
    im      = ax.imshow(x_true, cmap=self.cmap)
    divider = make_axes_locatable(ax)
    cax     = divider.append_axes("right", size="5%", pad=0.05)
    ax.set_title(r'$\vec{x}_{true}$')
    ax.axis('off')
    colorbar(im, cax=cax)
  
  def plot_b(self, ax):
    """
    plot the true image.
    """
    b       = self.b
    im      = ax.imshow(b, cmap=self.cmap)
    divider = make_axes_locatable(ax)
    cax     = divider.append_axes("right", size="5%", pad=0.05)
    ax.set_title(r'$\vec{b}$')
    ax.axis('off')
    colorbar(im, cax=cax)
  
  def solve(self, ax, rtol=1e-1, plot_iterations=False):
    """
    solve the system and update the plot as kaczmarz iterations progress.
    """
    n    = self.n
    A    = self.A
    vecb = self.vecb

    x  = zeros(n*n)
    X  = reshape(x,(n,n),'F')
    xn = x.copy()
   
    ion()
    im      = ax.imshow(X, cmap=self.cmap)
    divider = make_axes_locatable(ax)
    cax     = divider.append_axes("right", size="5%", pad=0.05)
    ax.set_title(r'$\vec{x}$')
    ax.axis('off')
    colorbar(im, cax=cax)
    draw()
    
    relative = 1e10
    while relative > rtol:
      
      for i,bi in enumerate(vecb):
        ai = A[i].toarray()[0]
        x += (bi - dot(ai, x))*ai / norm(ai)**2
        if plot_iterations:
          X = reshape(x,(n,n),'F')
          im.set_clim(X.min(), X.max())
          im.set_data(X)
          draw()
    
      X = reshape(x,(n,n),'F')
      im.set_clim(X.min(), X.max())
      im.set_data(X)
      draw()
      
      relative = norm(xn - x)
      print 'relative difference:', relative
      xn = x.copy()
    
    ioff()
    print 'Done!'


  def Xraymat(self, s, phi, N):
    """
    function A = Xraymat(s,phi,N)
    The program creates a sparse matrix A used in planar parallel beam
    X-ray tomography.
    
    The image area is the unit square [0,1]x[0,1]. The pixels are enumerated
    in the same order as in matrix indexing, i.e., first pixel is in the top
    left corner, the numbering proceeding columnwise.
    
    Each ray is parametrized by two numbers: phi is the angle between the
    line and vertical axis, phi = 0 corresponding to a ray going upwards,
    phi=-pi/2 going from left to right. The parameter s is the distance
    of the line from the midpoint of the image with sign, so that when phi=0
    and s>0, the line passes through the right half of the image. The line
    can be parametrized as
    
    x(t) = 0.5 + s*cos(phi) - t*sin(phi)
    y(t) = 0.5 + s*sin(phi) + t*cos(phi)
    
    Input: s   - a vector containing the s-parameters of the X-rays
           phi - a vector of same size as s containing the angles of the X-rays
           N   - integer giving pixel number per edge
    
    NOTE: entries of s must lie between -0.5/r and 0.5/r, where 
          r = max(|sin(phi)|, |cos(phi)|), otherwise there is no
          guarantee that they hit the image area and the programme may fail.
         
    Output: A  - sparse matrix of size (k,N*N), k being the length of the
                 vectors s and phi. The entry A(i,j) gives the intersection
                 length of the X-ray i with the pixel j.
    
    First version, Erkki Somersalo 10/2/2002
    """
    K    = len(s) 
    rows = []
    cols = []
    vals = []
    
    ss   = sin(phi) 
    cc   = cos(phi)
    p    = linspace(1,K,N) / float(K)
    
    print 'Assembling X-ray / tomography matrix.'
    for k in range(K):
      t_a = array([])
      x_a = array([])
      y_a = array([])
      # Finding intersection points with lines y = j/N, 0<=j<=N
      if abs(cc[k]) > 0:
        t   = (p - 0.5 - s[k]*ss[k]) / cc[k] 
        x   = 0.5 + s[k]*cc[k] - t*ss[k] 
        aux = (x>=0) & (x<=1)
        t_a = append(t_a, t[aux])
        x_a = append(x_a, x[aux])
        y_a = append(y_a, p[aux])
    
      # Finding intersection points with lines x = j/N, 0<=j<=N
      if abs(ss[k]) > 0:
        t   = (0.5 + s[k]*cc[k] - p) / ss[k]
        y   = 0.5 + s[k]*ss[k] + t*cc[k] 
        aux = (y>=0) & (y<=1)
        t_a = append(t_a, t[aux])
        x_a = append(x_a, p[aux])
        y_a = append(y_a, y[aux])
      
      # Sorting the intersection points according to increasing t
      I   = argsort(t_a)
      t_a = t_a[I]
      x_a = x_a[I]
      y_a = y_a[I]
      
      # Computing the intersection lengths and pixels.
      # If the X-ray passes from corner to corner of the pixel, the corner
      # coordinates appear twice. Discarding redundant intersections
      # corresponding to pairs giving zeros (or negligible) intersection length.
      n         = len(t_a) 
      lengths   =      t_a[1:n] - t_a[0:n-1] 
      xmids     = 0.5*(x_a[1:n] + x_a[0:n-1])
      ymids     = 0.5*(y_a[1:n] + y_a[0:n-1]) 
      iaux      = lengths > 0
      lengths   = lengths[iaux]
      xmids     = xmids[iaux]
      ymids     = ymids[iaux]
      indx      = ceil(N * xmids)
      indy      = ceil(N * ymids)
      rows     += (k*ones(len(indx))).tolist()
      cols     += ((indx-1)*N + (indy-1)).tolist()
      vals     += (lengths).tolist()
    
    A = csr_matrix((vals, (rows, cols)))
    print 'Done'
    return A



  

