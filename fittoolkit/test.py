# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 10:12:29 2026

@author: morit
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 19:12:09 2026

@author: morit
"""
from pathlib import Path
import sys
if str(Path(__file__).parent.parent.parent/'fancyfit/utils') not in sys.path:
   sys.path.append(str(Path(__file__).parent.parent.parent/'fancyfit/utils')) 
from auxiliary import FitFunctions
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import convolve
from fittoolkit import GlobalFit
from numpy.linalg import inv
from iminuit import Minuit as Mi

ff = FitFunctions()
ff.default()

thin_out = (10,20)
base = Path(__file__).parent/'Test Data'
# x = np.genfromtxt(Path(base)/'x.txt')[:-200]
# y = np.genfromtxt(Path(base)/'y.txt')[250:-50]
# Z = np.genfromtxt(Path(base)/'z.txt')[250:-50,:-200]
x = np.genfromtxt(Path(base)/'t.txt')[::thin_out[0]]
y = np.genfromtxt(Path(base)/'wl.txt')[::thin_out[1]]
Z = np.genfromtxt(Path(base)/'signal.txt')[::thin_out[1],::thin_out[0]]

    
funs=(ff.funs['exp_decay'].func,ff.funs['exp_decay'].func,ff.funs['exp_decay'].func)
names = (('x0','A','tau'),('x0','A','tau'),('x0','A','tau'))
p0 = ((-5,1,2),(-5,1,1000),(-5,1,2000))
pl = ((-5,1,1),(-5,1,900),(-5,1,1000))
pu = ((-5,1,40),(-5,1,1400),(-5,1,10000))
cp = ('x0')

GF = []
methods = ['migrad','L-BFGS-B','TNC','COBYLA','SLSQP','trust-constr','CG','Powell','BFGS',]
# probe = np.linspace(0.01,20,20)
# probe = np.arange(0,3,1)
# for i in probe:
for method in methods:
    s={
            'funs': funs,
            'parms': names,
            'p0': p0,
            'p_lower': pl,
            'p_upper': pu,
            'common_parms': cp,
            # 'iterations': i,
            'method': method,
            # 'tol': i,
            'scale': True,
            'print_level': 0,
            # 'simplex': False,
            # 'strategy': 1,
            # 'hesse': False,
             }
    
    gf = GlobalFit(x,y,Z,settings=s) 
    # print(gf.deviation[-1])
    # print(gf.m.values)
    GF.append(gf)
    
fig,ax =plt.subplots()
# ax.plot(probe,[gf.deviation[-1] for gf in GF],'x')
ax.plot(methods,[gf.deviation[-1] for gf in GF],'x')
# ax.set_title(f'iterations:{s["iterations"]}; ')






# #%%
# for n,gf in enumerate(GF):
#     print(f'{methods[n]}:{gf.deviation[-1]}')
    
# plt.plot(methods,[gf.deviation[-1] for gf in GF])

# #%%

# fig1,ax1 = plt.subplots()
# ax1.plot(x,Z[25,:])
# ax1.plot(x,gf.Z_fit[25,:],'-x')
# # ax1.plot(x,Z[10,:])
# # ax1.plot(x,gf.Z_fit[10,:],'-x')
# # ax1.plot(x,Z[2,:])
# # ax1.plot(x,gf.Z_fit[2,:],'-x')

# fig2,ax2 = plt.subplots()
# # ax2.plot(y,Z[:,2])
# # ax2.plot(y,gf.Z_fit[:,2],'-x')
# ax2.plot(y,Z[:,50])
# ax2.plot(y,gf.Z_fit[:,50],'-x')
# # ax2.plot(y,Z[:,250])
# # ax2.plot(y,gf.Z_fit[:,250],'-x')

# fig3,ax3 = plt.subplots()
# ax3.contourf(Z.T-gf.Z_fit.T)