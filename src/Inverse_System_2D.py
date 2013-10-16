from pylab          import *
from scipy.optimize import fminbound
from Inverse_System import *

class Inverse_System_2D(Inverse_System):

  def __init__(self, sig, x_true, A_ftn, err_lvl): 
    """
    class representing a system we wish to invert.
    """
    nx, ny  = shape(x_true)
    nx      = float(nx)
    ny      = float(ny)
    n       = nx * ny
    hx      = 1/nx
    hy      = 1/ny
    tx      = arange(0, 1, hx)
    ty      = arange(0, 1, hy)

    # A discritization :
    A1      = A_ftn(tx)
    A2      = A_ftn(ty)
    
    # Set up true solution x_true and data b = A*x_true + error :
    Ax      = dot(dot(A1, x_true), A2.T)
    sigma   = err_lvl/100.0 * norm(Ax) / sqrt(n)
    eta     = sigma * randn(nx, ny)
    b       = Ax + eta
   
    U1,S1,V1 = svd(A1)
    U2,S2,V2 = svd(A2)
    S        = tensordot(S2, S1, 0)
    UTb      = dot(dot(U1.T, b), U2)
    
    # 2D problems can only be filtered by Tikhonov regularization
    self.filt_type = 'Tikhonov'
    
    self.rng     = arange(0, 1, 0.1)
    self.n       = n
    self.nx      = nx
    self.ny      = ny
    self.tx      = tx
    self.ty      = ty
    self.A1      = A1
    self.A2      = A2
    self.x_true  = x_true
    self.Ax      = Ax
    self.err_lvl = err_lvl
    self.sigma   = sigma
    self.b       = b
    self.U1      = U1
    self.U2      = U2
    self.S1      = S1
    self.S2      = S2
    self.S       = S
    self.V1      = V1
    self.V2      = V2
    self.Vx      = dot(V1.T, dot(x_true, V2))
    self.UTb     = UTb

  def get_xfilt(self, alpha):
    """
    get the filtered x solution.
    """
    S      = self.S
    V1     = self.V1
    V2     = self.V2
    UTb    = self.UTb
    if self.filt_type == 'Tikhonov':
      dSfilt = S**2 / (S**2 + alpha) 
    else:
      dSfilt         = ones((self.nx, self.ny))
      dSfilt[alpha:] = 0.0
    x_filt = dot(V1.T, dot(dSfilt / S * UTb, V2))
    return x_filt
  
  def Lcurve(self, a):
    """
    Compute minus the curvature of the L-curve
    """
    return 0.0

  def plot_filt(self, ax, x_filt, alpha, tit):
    """
    plot the filtered solution.
    """
    st = tit + r' Filtered, $\alpha = %.2E$'
    ax.imshow(x_filt)
    ax.set_title(st % alpha)
  
  def plot_true(self, ax):
    """
    plot the true and blurred solution.
    """
    x_true = self.x_true
    ax.imshow(x_true)
    ax.set_title(r'$\vec{x}_{true}$')
  
  def plot_b(self, ax):
    """
    plot the true and blurred solution.
    """
    b = self.b
    ax.imshow(b)
    ax.set_title(r'$\vec{b}$')
  
  def plot_U_vectors(self, ax):
    """
    plot the first 8 orthogonal U vectors.
    """
    U = self.U
    t = self.t
    fig = figure(figsize=(12,7))
    axs = []
    for i in range(8):
      ax = fig.add_subplot(240 + (i+1))
      ax.plot(t, U[:,i], 'k-', lw=2.0)
      ax.grid()
      if i > 3: ax.set_xlabel(r'$t$')
      ax.set_title(r'$\vec{u}_{%i}$' % i)
    show()

  def plot_UTb_vectors(self, ax):
    """
    plot the singular vectors UTb on a log y-axis.
    """
    U  = self.U
    S  = self.S
    b  = self.b
    Ax = self.Ax
    t  = self.t

    ax.semilogy(t, S, 'r', lw=2, label=r'$\Sigma$')
    ax.semilogy(t, abs(dot(U.T, b)),    'ws', lw=2,
                label=r'$\vec{u}_i^T \cdot \vec{b}$')
    ax.semilogy(t, abs(dot(U.T, b)/S),  'w^', lw=2, 
                label=r'$\frac{\vec{u}_i^T \cdot \vec{b}}{\sigma}$')
    ax.semilogy(t, abs(dot(U.T, Ax)),   'ks', lw=2, 
                label=r'$\vec{u}_i^T \cdot A\vec{x}$')
    ax.semilogy(t, abs(dot(U.T, Ax)/S), 'k^', lw=2, 
                label=r'$\frac{\vec{u}_i^T \cdot A\vec{x}}{\sigma}$')
    ax.set_xlabel(r'$t$')
    leg = ax.legend(loc='upper left')
    leg.get_frame().set_alpha(0.5)
    ax.grid()

  def plot_VTx_variance(self, ax):
    """
    plot the variance values in V^T x_LS.
    """
    V       = self.V
    A       = self.A
    b       = self.b
    Ax      = self.Ax
    x_ls_no = solve(A,b)
    x_ls    = solve(A,Ax)

    ax.plot(dot(V.T, x_ls),    'r-',  label='clean', lw=2.0)
    ax.plot(dot(V.T, x_ls_no), 'ko-', label='noisy')
    ax.set_xlabel(r'$i$')
    ax.set_title(r'$\vec{v}_i^T \vec{x}_{LS}$')
    ax.grid()
    leg = ax.legend(loc='upper center')
    leg.get_frame().set_alpha(0.5)

  def plot_variance(self, ax):
    """
    plot the variance values sigma^2 / sigma_i^2.
    """
    sigma = self.sigma
    S     = self.S

    ax.plot(sigma/S**2, 'ko-',  label='variance', lw=2.0)
    ax.set_yscale('log')
    ax.set_title(r'Variance $\sigma^2/\sigma_i^2$')
    ax.set_xlabel(r'$i$')
    ax.grid()




