import sys
src_directory = '../src/'
sys.path.append(src_directory)

from pylab             import *
from functions         import *
from Inverse_System_1D import *

mpl.rcParams['font.family']     = 'serif'
mpl.rcParams['legend.fontsize'] = 'medium'

#===============================================================================
#  
#  1d image deblurring inverse problem with Dirichlet boundary conditions.
#

xi         = 0                  # begin of domain
xf         = 1                  # end of domain
n          = 80                 # number of nodes
sig        = 0.05               # desired SD of noise
PSF        = gaussian_PSF       # design matrix
x_true_ftn = wacky_thing        # true solution
err_lvl    = 2.0                # error level parameter

# object encapsulating the entire evaluative process on this system :
s = Inverse_System_1D(xi, xf, n, sig, err_lvl, x_true_ftn, PSF) 

rng = logspace(log10(1e-4), log10(1), 1000)
s.set_filt_type('GMRF', rng)

#===============================================================================
# query the user for a xfilter choice :
xfilt, alpha, tit = s.get_xfilt_choice(xi, xf)

# plotting
fig = figure(figsize=(13,9))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

# plot the true and filtered solutions :
s.plot_true(ax1)
s.plot_filt(ax2, xfilt, alpha, tit)

# get error functions over range :
s.calc_errors()

# find index of minimum and corresponding alphas :
idx1, a_min1 = s.find_min(s.fs)
idx2, a_min2 = s.find_min(s.GCVs)

# plot the errors on the same axes :
err_list = (s.fs,   s.GCVs)
idxs     = (idx1,   idx2)
alphas   = (a_min1, a_min2)
tits     = ('relative', 'GCV')
s.plot_errors(ax3, err_list, alphas, idxs, tits)

# Now compute the regularized solution for TSVD
xfilt = s.get_xfilt(a_min1)
s.plot_filt(ax4, xfilt, a_min1, 'Relative Error')

tight_layout()
#savefig('../doc/images/prb410GMRFdeblur.png', dpi=300)
show()

# plot all the errors on a loglog axis :
fig = figure()
ax  = fig.add_subplot(111)

s.plot_all_errors(ax)
tight_layout()
#savefig('../doc/images/prb410GMRFdeblur_error.png', dpi=300)
show()

