"""
Created on Fri Jul  3 19:12:09 2026

@author: morit
"""
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import erfc, erfcx

from fittoolkit import GlobalFit
from fittoolkit.utils.tools import convolve


def exp_decay(x,x0,A,tau):
    return A*np.exp(-(x-x0)/tau)*np.heaviside(x-x0,1)

def gauss(x,x0,A,s):
    return A/(s*np.sqrt(2*np.pi))*np.exp(-(x-x0)**2/(2*s**2))

def normgauss(x,x0,s):
    y = (s*np.sqrt(2*np.pi))*np.exp(-(x-x0)**2/(2*s**2))
    return y/max(y)

#%%

thin_out = (10,20)
base = Path(__file__).parent/'Test Data'
# x = np.genfromtxt(Path(base)/'x.txt')[:-200]
# y = np.genfromtxt(Path(base)/'y.txt')[250:-50]
# Z = np.genfromtxt(Path(base)/'z.txt')[250:-50,:-200]
x = np.genfromtxt(Path(base)/'t.txt')[::thin_out[0]]
y = np.genfromtxt(Path(base)/'wl.txt')[::thin_out[1]]
Z = np.genfromtxt(Path(base)/'signal.txt')[::thin_out[1],::thin_out[0]]

    
funs=(exp_decay,exp_decay,exp_decay)
names = (('x0','A','tau'),('x0','A','tau'),('x0','A','tau'))
p0 = ((-5,1,2),(-5,1,1000),(-5,1,2000))
pl = ((-5,1,1),(-5,1,900),(-5,1,1000))
pu = ((-5,1,40),(-5,1,1400),(-5,1,10000))
cp = ('x0')

IRF = gauss
irf_parms = ['x0','A','s']
irf_p0 = (1,1,1)
irf_p_lower = (-np.inf,-np.inf,-np.inf)
irf_p_upper = (np.inf,np.inf,np.inf)

GF = [1]
methods = ['migrad','L-BFGS-B','TNC','COBYLA','SLSQP','trust-constr','CG','Powell','BFGS',]
# probe = np.linspace(0.01,20,20)
# probe = np.arange(0,3,1)
# for i in probe:
for g in [1]:
    s={
            'funs': funs,
            'parms': names,
            'p0': p0,
            'p_lower': pl,
            'p_upper': pu,
            'common_parms': cp,
            # 'iterations': i,
            'method': 'Powell',
            # 'tol': i,
            'scale': True,
            'print_level': 0,
            # 'simplex': False,
            # 'strategy': 1,
            # 'hesse': False,
            'irf': IRF, #must be a function
            'irf_parms': irf_parms,
            'irf_p0': irf_p0,
            'irf_p_lower': irf_p_lower,
            'irf_p_upper': irf_p_upper,
             }
    
    gf = GlobalFit(x,y,Z,settings=s) 
    # print(gf.deviation[-1])
    # print(gf.m.values)
    GF.append(gf)
    
# fig,ax =plt.subplots()
# # ax.plot(probe,[gf.deviation[-1] for gf in GF],'x')
# ax.plot(methods,[gf.deviation[-1] for gf in GF],'x')
# # ax.set_title(f'iterations:{s["iterations"]}; ')






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


#%%

def exp_decay_conv_gauss(x, x0, A, tau, s):
    """
    Analytical convolution of  A*exp(-(x-x0)/tau)*H(x-x0)  with a unit-area
    Gaussian of width s (an exponentially modified Gaussian).

    Reduces to the bare exponential decay as s -> 0. A is the amplitude the
    un-convolved decay would have at x0.
    """
    z = np.asarray(x, dtype=float) - x0
    if tau <= 0 or s <= 0:
        raise ValueError("tau and s must be > 0")

    u = (s/tau - z/s) / np.sqrt(2)
    out = np.empty_like(z)

    # u >= 0 (rising edge / early times): erfc underflows while exp overflows.
    #   exp(s^2/2tau^2 - z/tau)*erfc(u) == exp(-z^2/2s^2)*erfcx(u)
    m = u >= 0
    out[m] = np.exp(-z[m]**2 / (2*s**2)) * erfcx(u[m])

    # u < 0 (tail): erfcx overflows, the direct form is well behaved.
    nm = ~m
    out[nm] = np.exp(s**2/(2*tau**2) - z[nm]/tau) * erfc(u[nm])

    return (A/2) * out

def exp_decay(x,x0,A,tau):
    return A*np.exp(-(x-x0)/tau)*np.heaviside(x-x0,1)

x = np.linspace(-10,100,1000)


kin = exp_decay(x,-2,1,100)
irf = normgauss(x,-2,0.1)

fig,ax = plt.subplots()
# ax.plot(x,kin)
# ax.plot(irf)
ax.plot(x,convolve(x,kin,lambda x:normgauss(x,0,1)))
ax.plot(x,exp_decay_conv_gauss(x,-2,1,100,1))
