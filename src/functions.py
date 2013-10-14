from pylab          import *
from scipy.optimize import fminbound
from scipy.linalg   import toeplitz

class inverse_system(object):

  def __init__(self, xi, xf, n, sig, x_true_ftn, A_ftn, err_lvl): 
    """
    class representing a system we wish to invert.
    """
    n       = float(n)
    omega   = xf - xi
    h       = omega/n
    t       = arange(xi, xf, h)

    # A discritization :
    A       = A_ftn(t)
    
    # Set up true solution x_true and data b = A*x_true + error :
    x_true  = x_true_ftn(t)
    Ax      = dot(A, x_true)
    sigma   = err_lvl/100 * norm(Ax) / sqrt(n)
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
    self.UTb     = UTb

  def set_filt_type(self, filt_type, rng=None):
    if filt_type not in ['TSVD', 'Tikhonov']:
      print 'please choose filt_type to be either "TSVD" or "Tikhonov".'
    else:
      if filt_type == 'TSVD': 
        self.rng = range(int(self.n))
      else: 
        if rng == None:
          print "specify a range for Tikhonov regulariztion"
          exit(1)
        else:
          self.rng = rng
      self.filt_type = filt_type

 
  def Lcurve(self, a):
    """
    Compute minus the curvature of the L-curve
    """
    dS2    = self.S**2 
    UTb    = self.UTb
    V      = self.V
    S      = self.S
    A      = self.A
    b      = self.b
     
    xalpha = dot(V, (S / (dS2 + a)) * UTb) 
    ralpha = dot(A, xalpha - b)
    xi     = norm(xalpha)**2 
    rho    = norm(ralpha)**2 
    
    # From Hansen 2010 -- seems to be incorrect.
    #zalpha = V*((dS./(dS2+alpha)).*(U.T*ralpha)) 
    #xi_p = (4/sqrt(alpha))*xalpha.T*zalpha 
    #calpha =   2*(xi*rho/xi_p) *...
    #           (alpha*xi_p*rho+2*sqrt(alpha)*xi*rho+alpha^2*xi*xi_p) / ...
    #           (alpha*xi^2+rho^2)^(3/2) 
    
    # From Vogel 2002. 
    xi_p   = sum(-2 * dS2 * UTb**2 / (dS2 + a)**3) 
    calpha = - ( (rho*xi) * (a*rho + a**2 * xi) + (rho*xi)**2 / xi_p ) / \
               ( rho**2 + a**2 * xi**2)**(3/2.) 
    return calpha
  
  def UPRE(self, a):
    """
    Unbiased Predictive Rist Estimator curve.
    """
    UTb   = self.UTb 
    sigma = self.sigma
    if self.filt_type == 'TSVD':
      phi_nu     = ones(self.n)
      phi_nu[a:] = 0.0
    
    elif self.filt_type == 'Tikhonov':
      dS2    = self.S**2
      phi_nu = dS2 / (dS2 + a)
    
    return sum( ((phi_nu - 1)*UTb)**2 + 2*sigma**2*phi_nu )
  
  def DP2(self, a):
    """
    Discrepancy Principe curve.
    """
    n     = self.n
    UTb   = self.UTb 
    sigma = self.sigma
    if self.filt_type == 'TSVD':
      phi_nu     = ones(self.n)
      phi_nu[a:] = 0.0
    
    elif self.filt_type == 'Tikhonov':
      dS2    = self.S**2
      phi_nu = dS2 / (dS2 + a)
    
    return (sum( ((phi_nu - 1)*UTb)**2) - n*sigma**2)**2
  
  def GCV(self, a):
    """
    Generalized Cross Validation curve.
    """
    n     = self.n
    UTb   = self.UTb 
    sigma = self.sigma
    if self.filt_type == 'TSVD':
      phi_nu     = ones(self.n)
      phi_nu[a:] = 0.0
    
    elif self.filt_type == 'Tikhonov':
      dS2    = self.S**2
      phi_nu = dS2 / (dS2 + a)
    
    return sum( ((phi_nu - 1)*UTb)**2) / (n - sum(phi_nu))**2
  
  def MSE(self, a):
    """
    Mean standard error function.
    """
    S      = self.S
    sigma  = self.sigma
    V      = self.V
    x_true = self.x_true
    if self.filt_type == 'Tikhonov':
      dSfilt = S**2 / (S**2 + a)
    else:
      dSfilt     = ones(self.n)
      dSfilt[a:] = 0.0
    return + sigma**2 * sum( (dSfilt / S)**2 ) \
           + sum( (1 - dSfilt)**2 * dot(V, x_true)**2 )
  
  def relative_error(self, a):
    """
    Relative error function.
    """
    S      = self.S
    V      = self.V
    UTb    = self.UTb
    x_true = self.x_true
    if self.filt_type == 'Tikhonov':
      dSfilt = S**2 / (S**2 + a)
    else:
      dSfilt     = ones(self.n)
      dSfilt[a:] = 0.0
    x_filt = dot(V.T, dSfilt / S * UTb) 
    return norm(x_filt - x_true) / norm(x_true)
  
  def calc_errors(self):
    """
    calculate all the errors that we have available.
    """
    filt_type = self.filt_type

    fs    = array([])
    MSEs  = array([])
    UPREs = array([])
    DP2s  = array([])
    GCVs  = array([])
    if filt_type != 'TSVD': Lcs = array([])
    for alpha in self.rng:
      fs    = append(fs,    self.relative_error(alpha))
      MSEs  = append(MSEs,  self.MSE(alpha))
      UPREs = append(UPREs, self.UPRE(alpha))
      DP2s  = append(DP2s,  self.DP2(alpha))
      GCVs  = append(GCVs,  self.GCV(alpha))
      if filt_type != 'TSVD': Lcs = append(Lcs, self.Lcurve(alpha))
    self.fs    = fs   
    self.MSEs  = MSEs 
    self.UPREs = UPREs
    self.DP2s  = DP2s 
    self.GCVs  = GCVs 
    if filt_type != 'TSVD': self.Lcs = Lcs

  def find_min(self, ftn):
    """
    find and return index of min(fx), and this minimum
    """
    idx   = where(ftn == min(ftn))[0][0]
    a_min = self.rng[idx]
    return idx, a_min

  def get_xfilt_choice(self, xi, xf):
    """
    present a list of alpha-choosing optimization choices and give 
    the filtered solution back.
    """
    s =   ' Enter 0 to enter k, 1 for UPRE, 2 for GCV, ' \
        + '3 for DP, or 4 for L-curve: '
    param_choice = int(raw_input(s))
    
    if param_choice == 0:
      alpha = float(raw_input('alpha = '))
      tit = 'k'
    elif param_choice == 1:
      RegParam_fn = lambda a : self.UPRE(a)
      tit = 'UPRE'
    elif param_choice == 2:
      RegParam_fn = lambda a : self.GCV(a) 
      tit = 'GCV'
    elif param_choice == 3:
      RegParam_fn = lambda a : self.DP2(a) 
      tit = 'DP'
    elif param_choice == 4 and self.filt_type == 'Tikhonov':
      RegParam_fn = lambda a : self.Lcurve(a)
      tit = 'L-curve'
    elif param_choice == 4 and self.filt_type == 'TSVD':
      print 'ERROR: cannot use L-curve with TSVD'
      exit(1)
   
    # only compute the alpha if you did not explicitly state it :
    if self.filt_type == 'Tikhonov' and param_choice != 0: 
      alpha = fminbound(RegParam_fn, xi, xf)
    elif self.filt_type == 'TSVD' and param_choice != 0:
      alpha = fminbound(RegParam_fn, 0, self.n)
    
    # Now compute the regularized solution for TSVD
    x_filt = self.get_xfilt(alpha)
    return x_filt, alpha, tit

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
  
  def plot_errors(self, ax, err_list, alphas, idxs, err_tits):
    """
    plot the errors :
    """
    fs   = self.fs
    MSEs = self.MSEs

    # find index of minimum :
    rng  = self.rng
    
    if self.filt_type == 'Tikhonov':
      ls    = '-'
      lab1  = r'' + err_tits[0] + ': %.2E'
      lab2  = r'' + err_tits[1] + ': %.2E'
      ax.loglog(rng, err_list[0],    ls, lw=2.0, label=lab1 % alphas[0])
      ax.loglog(rng, err_list[1],    ls, lw=2.0, label=lab2 % alphas[1])
      ax.loglog(alphas[0], err_list[0][idxs[0]], 'd', color='#3d0057', 
                markersize=10, label=r'min{$\alpha$}')
      ax.loglog(alphas[1], err_list[1][idxs[1]], 'd', color='#3d0057', 
                markersize=10)
    elif self.filt_type == 'TSVD':
      ls    = 'o'
      lab1  = r'' + err_tits[0] + ': %.f'
      lab2  = r'' + err_tits[1] + ': %.f'
      ax.semilogy(rng, err_list[0],    ls, lw=2.0, label=lab1 % alphas[0])
      ax.semilogy(rng, err_list[1],    ls, lw=2.0, label=lab2 % alphas[1])
      ax.semilogy(alphas[0], err_list[0][idxs[0]], 'd', color='#3d0057', 
                  markersize=10, label=r'min{$\alpha$}')
      ax.semilogy(alphas[1], err_list[1][idxs[1]], 'd', color='#3d0057', 
                  markersize=10)

    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'ERROR$(\alpha)$')
    ax.set_title(r'Errors')
    leg = ax.legend()
    leg.get_frame().set_alpha(0.5)
    ax.grid()
  
  def plot_all_errors(self, ax):
    """
    plot a list of error values <errors> over range <rng> with 
    corresponding titles <tits> to axes object <ax>.
    """
    rng    = self.rng

    if self.filt_type == 'Tikhonov':
      ls       = '-'
      fmt      = '.2E'
      errors   = [self.fs, self.MSEs, self.UPREs, 
                  self.DP2s, self.GCVs, self.Lcs]
      err_tits = ['relative', 'MSE', 'UPRE', 'DP2', 'GCV', 'L-curve']
      for e, t in zip(errors, err_tits):
        if t == 'L-curve':
          idx, a = self.find_min(-e)
        else:
          idx, a = self.find_min(e)
        st = r'' + t + ': %' + fmt
        ax.loglog(rng, e, ls, lw=2.0, label=st % a)
        ax.plot(a, e[idx], 'd', color='#3d0057', markersize=10)
      leg = ax.legend(loc='lower left')
      leg.get_frame().set_alpha(0.5)
    
    elif self.filt_type == 'TSVD':
      ls       = 'o'
      fmt      = '.f'
      errors   = [self.fs, self.MSEs, self.UPREs, self.DP2s, self.GCVs]
      err_tits = ['relative', 'MSE', 'UPRE', 'DP2', 'GCV']
      for e, t in zip(errors, err_tits):
        if t == 'L-curve':
          idx, a = self.find_min(-e)
        else:
          idx, a = self.find_min(e)
        st = r'' + t + ': %' + fmt
        ax.semilogy(rng, e, ls, lw=2.0, label=st % a)
        ax.plot(a, e[idx], 'd', color='#3d0057', markersize=10)
      leg = ax.legend(loc='upper right')
      leg.get_frame().set_alpha(0.5)
   
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'ERROR$(\alpha)$')
    ax.set_title(r'Errors')
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


def wacky_thing(t):
  """
  crazy function for x_true.
  """
  wack =  + 50 * (  0.75 * ((0.1 < t) & (t < 0.25)) \
                  + 0.25 * ((0.3 < t) & (t < 0.32)) \
                  + ((0.5 < t) & (t < 1)) * sin(2*pi*t)**4 )
  return wack / norm(wack)

def PSF(t):
  """
  PSF reconstruction x_true.
  """
  h      = (max(t) - min(t)) / len(t) 
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

def PSF2(t):
  kernel  = zeros(len(t))
  low     = (-1/10. <= t) & (t <= 0)
  high    = (0      <  t) & (t <  1/10.)
  kernel[low]  =  100*t[low]  + 10
  kernel[high] = -100*t[high] + 10
  return kernel

            

def gaussian_kernel(t):
  """
  Descritization of the gaussian kernel function.
  """
  n      = len(t)
  h      = (max(t) - min(t)) / n 
  sig    = 0.05
  kernel = 1 / (sqrt(pi) * sig) * exp(-(t-h/2)**2 / sig**2)
  A      = h * toeplitz(kernel)
  return A

def integral_op(t):
  """
  Descritization of the integral operator.
  """
  n = len(t)
  h = (max(t) - min(t)) / n 
  A = h * tril(ones((n,n)))
  return A



