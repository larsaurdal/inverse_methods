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
#  PSF reconstruction inverse problem with Dirichlet boundary conditions.
#

xi         = 0                  # begin of domain
xf         = 1                  # end of domain
n          = 80                 # number of nodes
sig        = 0.05               # desired SD of noise
x_true_ftn = PSF                # true solution
err_lvl    = 2.0                # error level parameter

# object encapsulating the entire evaluative process on this system :
s = Inverse_System_1D(xi, xf, n, sig, err_lvl, x_true_ftn, None, recon=True)

rng = logspace(log10(1e-5), log10(1), 1000)
s.set_filt_type('IGMRF', rng, igmrf_iter=10)

#===============================================================================
# query the user for a xfilter choice :
xfilt, err, alpha = s.IGMRF_LD(s.GCV)

# plotting
fig = figure(figsize=(12,4))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

# plot the true and filtered solutions :
s.plot_true(ax1)
s.plot_filt(ax2, xfilt, alpha, 'GCV')

# find index of minimum and corresponding alpha :
idx, a_min = s.find_min(err)

# plot the error :
s.plot_error(ax3, err, a_min, idx, 'GCV')

tight_layout()
savefig('prb410IGMRF.png', dpi=300)
show()


