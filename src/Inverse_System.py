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
from pylab          import *
from scipy.optimize import fminbound

class Inverse_System(object):

  def __init__(self): 
    """
    class representing a system we wish to invert.
    """
    self.rng     = None 
    self.omega   = None 
    self.n       = None 
    self.h       = None 
    self.t       = None 
    self.A       = None 
    self.x_true  = None 
    self.x_ls    = None 
    self.Ax      = None 
    self.err_lvl = None 
    self.sigma   = None 
    self.b       = None 
    self.U       = None 
    self.S       = None 
    self.V       = None 
    self.Vx      = None
    self.UTb     = None
    self.D       = None
    self.L       = None 
  
  def set_filt_type(self, filt_type, rng=None):
    if filt_type not in ['TSVD', 'Tikhonov', 'GMRF']:
      print 'please choose filt_type to be either' +\
            ' "TSVD", "Tikhonov", or "GMRF".'
    else:
      if filt_type == 'TSVD': 
        self.rng = range(int(self.n))
      elif filt_type == 'Tikhonov' or filt_type == 'GMRF': 
        if rng == None:
          print "specify a range for Tikhonov or GMRF regulariztion"
          exit(1)
        else:
          self.rng = rng
      self.filt_type = filt_type
  
  def get_xfilt(self, alpha):
    pass

  def get_ralpha(self, alpha):
    pass
 
  def Lcurve(self, a):
    """
    Compute minus the curvature of the L-curve
    """
    dS2    = self.S**2 
    UTb    = self.UTb
    S      = self.S
    L      = self.L
    
    if self.filt_type == 'Tikhonov':
      phi_nu = S**2 / (dS2 + a)**3
    elif self.filt_type == 'GMRF':
      phi_nu = diag(a * L)
      phi_nu = S**2 / (dS2 + a)**3
    
    xalpha = self.get_xfilt(a) 
    ralpha = self.get_ralpha(a, xalpha)
    xi     = norm(xalpha)**2 
    rho    = norm(ralpha)**2 
    
    # From Vogel 2002. 
    xi_p   = sum(-2 * phi_nu * UTb**2) 
    calpha = - ( (rho*xi) * (a*rho + a**2 * xi) + (rho*xi)**2 / xi_p ) / \
               ( rho**2 + a**2 * xi**2)**(3/2.) 
    return calpha
  
  def UPRE(self, a):
    """
    Unbiased Predictive Rist Estimator curve.
    """
    UTb   = self.UTb 
    sigma = self.sigma
    L     = self.L
    if self.filt_type == 'TSVD':
      phi_nu     = ones(self.n)
      phi_nu[a:] = 0.0
    elif self.filt_type == 'Tikhonov':
      dS2    = self.S**2
      phi_nu = dS2 / (dS2 + a)
    elif self.filt_type == 'GMRF':
      phi_nu = diag(a * L)
    return sum( ((phi_nu - 1)*UTb)**2 + 2*sigma**2*phi_nu )
  
  def DP2(self, a):
    """
    Discrepancy Principe curve.
    """
    n     = self.n
    UTb   = self.UTb 
    sigma = self.sigma
    L     = self.L
    if self.filt_type == 'TSVD':
      phi_nu     = ones(self.n)
      phi_nu[a:] = 0.0
    elif self.filt_type == 'Tikhonov':
      dS2    = self.S**2
      phi_nu = dS2 / (dS2 + a)
    elif self.filt_type == 'GMRF':
      phi_nu = diag(a * L)
    return (sum( ((phi_nu - 1)*UTb)**2) - n*sigma**2)**2
  
  def GCV(self, a):
    """
    Generalized Cross Validation curve.
    """
    n     = self.n
    UTb   = self.UTb 
    sigma = self.sigma
    L     = self.L
    A     = self.A
    b     = self.b
    if self.filt_type == 'TSVD':
      phi_nu     = ones(self.n)
      phi_nu[a:] = 0.0
      GCV        = sum( ((phi_nu - 1)*UTb)**2) / (n - sum(phi_nu))**2
    elif self.filt_type == 'Tikhonov':
      dS2        = self.S**2
      phi_nu     = dS2 / (dS2 + a)
      GCV        =  sum( ((phi_nu - 1)*UTb)**2) / (n - sum(phi_nu))**2
    elif self.filt_type == 'GMRF':
      ATA        = dot(A.T, A)
      Aa         = dot(inv(ATA + a*L), A.T)
      reg_mat    = dot(A, Aa)
      GCV        = n * norm(dot(reg_mat, b) - b)**2 / (n - trace(reg_mat))**2
    return GCV
  
  def MSE(self, a):
    """
    Mean standard error function.
    """
    S      = self.S
    sigma  = self.sigma
    Vx     = self.Vx
    x_true = self.x_true
    L      = self.L
    if self.filt_type == 'Tikhonov':
      dSfilt = S**2 / (S**2 + a)
    elif self.filt_type == 'TSVD':
      dSfilt     = ones(self.n)
      dSfilt[a:] = 0.0
    elif self.filt_type == 'GMRF':
      dSfilt = diag(a * L)
    return sigma**2 * sum((dSfilt / S)**2) + sum((1 - dSfilt)**2 * Vx**2)
  
  def relative_error(self, a):
    """
    Relative error function.
    """
    S      = self.S
    UTb    = self.UTb
    x_true = self.x_true
    L      = self.L
    if self.filt_type == 'Tikhonov':
      dSfilt = S**2 / (S**2 + a)
    elif self.filt_type == 'TSVD':
      dSfilt     = ones(self.n)
      dSfilt[a:] = 0.0
    elif self.filt_type == 'GMRF':
      dSfilt = diag(a * L)
    x_filt = self.get_xfilt(a) 
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
    f_type       = self.filt_type    

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
    elif param_choice == 4 and (f_type == 'Tikhonov' or f_type == 'GMRF'):
      RegParam_fn = lambda a : self.Lcurve(a)
      tit = 'L-curve'
    elif param_choice == 4 and f_type == 'TSVD':
      print 'ERROR: cannot use L-curve with TSVD'
      exit(1)
   
    # only compute the alpha if you did not explicitly state it :
    if f_type == 'TSVD' and param_choice != 0:
      alpha = fminbound(RegParam_fn, 0, self.n)
    elif (f_type == 'Tikhonov' or f_type == 'GMRF') and param_choice != 0: 
      alpha = fminbound(RegParam_fn, xi, xf)
    
    # Now compute the regularized solution for TSVD
    x_filt = self.get_xfilt(alpha)
    return x_filt, alpha, tit

  def get_xfilt(self, alpha):
    """
    get the filtered x solution.
    """
    pass

  def plot_errors(self, ax, err_list, alphas, idxs, err_tits):
    """
    plot the errors :
    """
    fs   = self.fs
    MSEs = self.MSEs

    # find index of minimum :
    rng  = self.rng
    
    if self.filt_type == 'Tikhonov' or self.filt_type == 'GMRF':
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
  
  def plot_all_errors(self, ax, leg_loc='upper left'):
    """
    plot a list of error values <errors> over range <rng> with 
    corresponding titles <tits> to axes object <ax>.
    """
    rng    = self.rng

    if self.filt_type == 'Tikhonov' or self.filt_type == 'GMRF':
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
      leg = ax.legend(loc=leg_loc)
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
      leg = ax.legend(loc=leg_loc)
      leg.get_frame().set_alpha(0.5)
   
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'ERROR$(\alpha)$')
    ax.set_title(r'Errors')
    ax.grid()


