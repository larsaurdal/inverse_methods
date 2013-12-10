import sys
src_directory = '../src/'
sys.path.append(src_directory)

from pylab      import *
from Tomography import Tomography 

#  
#  Tomography Inverse Problem
#
## Generate data from Shepp-Logan phantom
n         = 200 
ntheta    = 200 
nz        = 100

s = Tomography(n, ntheta, nz, err_lvl=2.0, cmap='gray')

fig = figure(figsize=(13,4.5))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

tight_layout(pad=4.0)
s.plot_true(ax1)
s.plot_b(ax2)

# solve the problem :
s.solve(ax3, rtol=1e00, plot_iterations=False)
#savefig('images/prb314.png', dpi=300)
show()
