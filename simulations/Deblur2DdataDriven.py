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
                      per_BC_pad=True, restrict_dom=(100,228), cmap='Greys_r')

tau = 0.9
s.set_filt_type('Landweber', tau)

#===============================================================================
# plotting
img_dir = '/home/pf4d/exmortis223@gmail.com/2013 Autumn/M 514 - Inverse Methods/homework/05/doc/images/'

fig = figure(figsize=(12,4))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

# plot the true and blurred vectors :
s.plot_true(ax1)
s.plot_b(ax2)

# find index of minimum and corresponding alphas :
a_man = 0.0011

# Now compute the regularized solution for TSVD
xfiltMan  = s.get_xfilt(a_man)

# plot both the solutions :
s.plot_filt(ax3, xfiltMan,  a_man, 'manual', tau=True)
tight_layout()
#savefig(img_dir + 'prb312.png', dpi=300)
show()



