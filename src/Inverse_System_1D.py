from pylab          import *
from scipy.optimize import fminbound
from Inverse_System import *
from functions      import descritize_PSF_kernel as d_psf
from functions      import descritize_integral   as d_int

class Inverse_System_1D(Inverse_System):

  def __init__(self, xi, xf, n, sig, err_lvl, x_true_ftn, PSF, recon=False):
    """
    class representing a system we wish to invert.
    """
    n       = float(n)
    omega   = xf - xi
    h       = omega/n
    t       = arange(xi, xf, h)

    # A discritization :
    # A discritization :
    if not recon:
      A     = d_psf(t, PSF(t, sig=sig))
    else:
      A     = d_int(t)
    
    # Set up true solution x_true and data b = A*x_true + error :
    x_true  = x_true_ftn(t)
    Ax      = dot(A, x_true)
    sigma   = err_lvl/100.0 * norm(Ax) / sqrt(n)
    eta     = sigma * randn(n, 1)
    b       = Ax + eta.T[0]
    x_ls    = solve(A,b)
   
    U,S,V   = svd(A)
    UTb     = dot(U.T, b)
    
    # by default, filter by Tikhonov parameterization
    self.filt_type = 'Tikhonov'
    
    self.rng     = arange(0, 1, 0.1)
    self.omega   = omega 
    self.n       = n
    self.h       = h
    self.t       = t
    self.A       = A
    self.x_true  = x_true
    self.x_ls    = x_ls
    self.Ax      = Ax
    self.err_lvl = err_lvl
    self.sigma   = sigma
    self.b       = b
    self.U       = U
    self.S       = S
    self.V       = V
    self.Vx      = dot(V, x_true)
    self.UTb     = UTb

  def get_xfilt(self, alpha):
    """
    get the filtered x solution.
    """
    S      = self.S
    V      = self.V
    UTb    = self.UTb
    if self.filt_type == 'Tikhonov':
      dSfilt = S**2 / (S**2 + alpha) 
    else:
      dSfilt         = ones(self.n)
      dSfilt[alpha:] = 0.0
    x_filt = dot(V.T, dSfilt / S * UTb)
    return x_filt

  def plot_filt(self, ax, x_filt, alpha, tit):
    """
    plot the filtered solution.
    """
    if self.filt_type == 'Tikhonov':
      st = tit + r' Filtered, $\alpha = %.2E$'
    elif self.filt_type == 'TSVD':
      st = tit + r' Filtered, $\alpha = %.f$ '
    
    t      = self.t
    x_true = self.x_true

    ax.plot(t, x_true, 'k-', label='true',        lw=1.5)
    ax.plot(t, x_filt, 'r-', label=r'$x_{filt}$', lw=1.5)
    ax.set_title(st % alpha)
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$x$')
    leg = ax.legend(loc='upper left')
    leg.get_frame().set_alpha(0.5)
    ax.grid()
  
  def plot_two_filt(self, ax, x_filts, alphas, tits):
    """
    plot the filtered solution.
    """
    if self.filt_type == 'Tikhonov':
      st1 = tits[0] + r', $\alpha = %.2E$'
      st2 = tits[1] + r', $\alpha = %.2E$'
    elif self.filt_type == 'TSVD':
      st1 = tits[0] + r', $\alpha = %.f$'
      st2 = tits[1] + r', $\alpha = %.f$'
    
    t      = self.t
    x_true = self.x_true

    ax.plot(t, x_true,     'k-', label='true',          lw=1.5)
    ax.plot(t, x_filts[0], 'r-', label=st1 % alphas[0], lw=1.5)
    ax.plot(t, x_filts[1], 'g-', label=st2 % alphas[1], lw=1.5)
    ax.set_title(r'Filtered Solutions')
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$x$')
    leg = ax.legend(loc='upper left')
    leg.get_frame().set_alpha(0.5)
    ax.grid()
  
  def plot_ls(self, ax):
    """
    plot the least-squares solution.
    """
    t      = self.t
    x_true = self.x_true
    x_ls   = self.x_ls

    ax.plot(t, x_true, 'k-', label='true', lw=1.5)
    ax.plot(t, x_ls,   'r-', label=r'$\vec{x}_{LS}$', lw=1.5)
    ax.set_title(r'Noisy $A\vec{x} = \vec{b}$')
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$x$')
    leg = ax.legend(loc='upper left')
    leg.get_frame().set_alpha(0.5)
    ax.grid()
  
  def plot_true(self, ax):
    """
    plot the true and blurred solution.
    """
    t      = self.t
    x_true = self.x_true
    b      = self.b

    ax.plot(t, x_true, 'k-', label='true image', lw=1.5)
    ax.plot(t, b,      'ro', label='blurred')
    ax.set_title(r'True')
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$x$')
    leg = ax.legend(loc='upper left')
    leg.get_frame().set_alpha(0.5)
    ax.grid()
  
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




