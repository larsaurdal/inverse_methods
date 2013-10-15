import sys
src_directory = '../src/'
sys.path.append(src_directory)

from pylab             import *
from Inverse_System_1D import *
from functions         import *

mpl.rcParams['font.family']     = 'serif'
mpl.rcParams['legend.fontsize'] = 'medium'

#===============================================================================
#  
#  PSF reconstruction inverse problem with Dirichlet boundary conditions.
#

xi         = -0.4
xf         =  0.4
n          = 80
sig        = 0.05
A_ftn      = integral_op
x_true_ftn = PSF2
err_lvl    = 2.0

# range for plotting errors :
s = Inverse_System_1D(xi, xf, n, sig, x_true_ftn, A_ftn, err_lvl)

rng = logspace(log10(1e-10), log10(10), 1000)
s.set_filt_type('Tikhonov', rng)

#===============================================================================
# plotting
img_dir = '/home/pf4d/exmortis223@gmail.com/UM/2013 Autumn/M 514 - Inverse Methods/homework/04/doc/images/'

fig = figure(figsize=(12,5))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

# plot the true and filtered solutions :
s.plot_true(ax1)

# get error functions over range :
s.calc_errors()

# find index of minimum and corresponding alphas :
idx1, a_min1 = s.find_min(s.GCVs)
idx2, a_min2 = s.find_min(-s.Lcs)

# plot the errors on the same axes :
err_list = (s.GCVs, s.Lcs)
idxs     = (idx1, idx2)
alphas   = (a_min1, a_min2)
tits     = ('GCV', 'L-curve')
s.plot_errors(ax2, err_list, alphas, idxs, tits)

# Now compute the regularized solution for TSVD
xfiltUPRE = s.get_xfilt(a_min1)
xfiltDP   = s.get_xfilt(a_min2)

# plot both the solutions :
xfilts = (xfiltUPRE, xfiltDP)
s.plot_two_filt(ax3, xfilts, alphas, tits)
tight_layout()
#savefig(img_dir + 'prb29b.png', dpi=300)
show()

# plot all the errors on a loglog axis :
fig = figure()
ax  = fig.add_subplot(111)

s.plot_all_errors(ax)
#savefig(img_dir + 'prb29b_error.png', dpi=300)
show()



