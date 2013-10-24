import sys
src_directory = '../src/'
sys.path.append(src_directory)

from pylab             import *
from Inverse_System_2D import *
from functions         import *
from scipy.io          import loadmat

mpl.rcParams['font.family']     = 'serif'
mpl.rcParams['legend.fontsize'] = 'medium'

#===============================================================================
#  
#  2d image deblurring inverse problem with separable kernel.
#

f_true = loadmat('../data/GaussianBlur440_normal.mat')['f_true']
x_true = 100*f_true / f_true.max()
nx,ny  = shape(x_true)

sig        = 2.0 / nx
PSF        = gaussian_PSF_2D
err_lvl    = 2.0

# range for plotting errors :
s = Inverse_System_2D(sig, err_lvl, x_true, PSF, per_BC=True, per_t=0.5,
                      restrict_dom=(100,228), cmap='Greys_r')

rng = logspace(log10(1e-15), log10(10), 1000)
s.set_filt_type('Tikhonov', rng)

#===============================================================================
# plotting
img_dir = '/home/pf4d/exmortis223@gmail.com/UM/2013 Autumn/M 514 - Inverse Methods/homework/05/doc/images/'

fig = figure(figsize=(13,9))
ax1 = fig.add_subplot(231)
ax2 = fig.add_subplot(232)
ax3 = fig.add_subplot(233)
ax4 = fig.add_subplot(234)
ax5 = fig.add_subplot(235)
ax6 = fig.add_subplot(236)

# plot the true and blurred vectors :
s.plot_true(ax1)
s.plot_b(ax2)
# get error functions over range :
s.calc_errors()

# find index of minimum and corresponding alphas :
idx1, a_min1 = s.find_min(s.UPREs)
idx2, a_min2 = s.find_min(s.DP2s)
idx3, a_min3 = s.find_min(s.GCVs)
idx4, a_min4 = s.find_min(s.MSEs)
a_man = 0.0011

# Now compute the regularized solution for TSVD
xfiltUPRE = s.get_xfilt(a_min1)
xfiltDP2  = s.get_xfilt(a_min2)
xfiltGCV  = s.get_xfilt(a_min3)
xfiltMSE  = s.get_xfilt(a_min4)
xfiltMan  = s.get_xfilt(a_man)

# plot both the solutions :
s.plot_filt(ax3, xfiltUPRE, a_min1, 'UPRE')
s.plot_filt(ax4, xfiltDP2,  a_min2, 'DP2')
#s.plot_filt(ax5, xfiltGCV,  a_min3, 'GCV')
s.plot_filt(ax6, xfiltMSE,  a_min4, 'MSE')
s.plot_filt(ax5, xfiltMan,  a_man, 'manual')
tight_layout()
savefig(img_dir + 'prb38b.png', dpi=300)
show()

# plot all the errors on a loglog axis :
fig = figure()
ax  = fig.add_subplot(111)
s.plot_all_errors(ax)
savefig(img_dir + 'prb38b_error.png', dpi=300)
show()




