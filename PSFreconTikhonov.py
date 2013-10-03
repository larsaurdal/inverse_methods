from pylab     import *
from functions import *

mpl.rcParams['font.family']     = 'serif'
mpl.rcParams['legend.fontsize'] = 'medium'

#===============================================================================
#  
#  PSF reconstruction inverse problem with Dirichlet boundary conditions.
#

xi         = 0
xf         = 1
n          = 80
sig        = 0.05
A_ftn      = integral_op
x_true_ftn = PSF
err_lvl    = 2.0
s          = inverse_system(xi, xf, n, sig, x_true_ftn, A_ftn, err_lvl)

#===============================================================================

# query the user for a xfilter choice :
xfilt, alpha, tit = s.get_xfilt_choice()

# plotting
fig = figure(figsize=(13,9))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

# plot the true and filtered solutions :
s.plot_true(ax1)
s.plot_filt(ax2, xfilt, alpha, tit)

# range for plotting errors :
rng = logspace(log10(1e-5), log10(1), 1000)

# get error functions over range :
fs, MSEs, UPREs, DP2s, GCVs, Lcs = s.calc_errors(rng)

# find index of minimum and corresponding alphas :
idxf, a_minf = s.find_min(rng, fs)
idxm, a_minm = s.find_min(rng, MSEs)

# plot the errors on the same axes :
s.plot_errors(ax3, rng, fs, MSEs)

# Now compute the regularized solution for TSVD
xfilt = s.get_xfilt(a_minf)
s.plot_filt(ax4, xfilt, a_minf, 'Relative Error')

savefig('../doc/images/prb25psf.png', dpi=300)
show()

# plot all the errors on a loglog axis :
fig = figure()
ax  = fig.add_subplot(111)

errors = [UPREs, DP2s, GCVs, Lcs]
tits   = ['UPRE', 'DP2', 'GCV', 'L-curve']
s.plot_error_list(ax, rng, errors, tits)
savefig('../doc/images/prb25psf_error.png', dpi=300)
show()
