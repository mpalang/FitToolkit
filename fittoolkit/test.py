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
if str(Path(__file__).parent.parent) not in sys.path:
   sys.path.append(str(Path(__file__).parent.parent)) 
from utils.auxiliary import FitFunctions
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import convolve
from fittoolkit import GlobalFit
from numpy.linalg import inv
from iminuit import Minuit as Mi

ff = FitFunctions()
ff.default()

base = Path(__file__).parent/'Test Data'
x = np.genfromtxt(Path(base)/'x.txt')[:-200]
y = np.genfromtxt(Path(base)/'y.txt')[250:-50]
Z = np.genfromtxt(Path(base)/'z.txt')[250:-50,:-200]

    
funs=(ff.funs['exp_decay'].func,ff.funs['exp_decay'].func,ff.funs['exp_decay'].func)
names = (('x0','A','tau'),('x0','A','tau'),('x0','A','tau'))
p0 = ((0,1,1),(0,1,1),(0,1,1))
pl = ((-1,1,1),(-1,1,1),(-1,1,1))
pu = ((1,1,1000),(1,1,1000),(1,1,1000))
cp = ('x0')


settings={
        'funs': funs,
        'parms': names,
        'p0': p0,
        'p_lower': pl,
        'p_upper': pu,
        'common_parms': cp,
         }

gf = GlobalFit(x,y,Z,settings=settings) 
pfit = gf.p

fig,ax = plt.subplots()
ax.plot(x,Z[40,:])
ax.plot(x,gf.Z_fit[40,:])