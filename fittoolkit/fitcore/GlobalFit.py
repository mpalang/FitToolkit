# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 11:29:24 2023

@author: work

WL: Wavelength 1D array
dt: Delay times 1D array
Z: Data Values as 2D array with shape (len(WL),len(dt))

"""
import numpy as np
from numpy.linalg import inv
from iminuit import Minuit as Mi


class GlobalFit:
    """
    Creactes global fit Object to fit 2D spectroscopic Data and return DADS.
    Possible Fit Methods:
        'L-BFGS-B','TNC','COBYLA','SLSQP','trust-constr','CG','Powell','BFGS',
        #Additional Information required: 'trust-krylov','trust-ncg','dogleg','Newton-CG','trust-exact'
    """   
    # =============================================================================
    #     Initialize the global_fit object
    # =============================================================================   
    def __init__(self,x,y,Z,name="Data",settings=None):
        
        self.set={}
        self.standard_set()
        if settings:
            self.set_update(**settings)
        else:
            raise ValueError('settings must not be empty')
        
        self.no_comps = len(self.set['funs'])
        self.y=y
        self.x=x # axis along which functions are fit
        self.Z=Z # axis along which DADS are determined
        
        self.DADS=np.zeros((len(y),self.no_comps))
        if self.set['fixed DADS']:
            for n,i in self.set['fixed DADS']: #if ever using this maybe theres a better way to just make numpy array
                self.DADS[:,n]=i
                
        self.D=np.zeros((self.no_comps,len(x)))
        self.Z_fit= np.zeros(self.Z.shape)
        self.result='No Fit Available'
        
        self.check_data()
        self.fit()

            
    # =============================================================================
    # Settings and data wrangling
    # =============================================================================
    def standard_set(self):
        """
        Returns
        -------
        None.

        """
        self.set={
            
                'funs':None, # is a list of functions.
                'parms':[], # is a list of tuples with parm names
                'p0':[], #is a list of tuples of guesses
                'p_lower':[], #is a list of tuples of lower boundaries. Can be None.
                'p_upper':[], #is a list of tuples of upper boundaries. Can be None.
                'common_parms': [], # Tuple of parm names that are common between different functions.
                'method': None,
                'scan': False,
                'tol':1e-3,
                'errors':None,
                'strategy':None,
                'print_level':None,
                'iterations':2,
                'fixed DADS': False,
                    }
    
    def check_data(self):
        if self.Z.shape != (np.squeeze(self.y.shape),np.squeeze(self.x.shape)):
            raise ValueError()
        
        # bring parameters to same magnitude for better fit:
        scaling_factors=[]
        for n,parm in enumerate(self.p0):
            factor = 10**(-np.floor(np.log10(parm)))
            if factor and not np.isinf(factor):
                scaling_factors.append(factor)
            else:
                scaling_factors.append(1)
            self.p0[n] = self.p0[n]*scaling_factors[n]
            self.p_lower[n] = self.p_lower[n]*scaling_factors[n]
            self.p_upper[n] = self.p_upper[n]*scaling_factors[n]
        self.scaling_factors = dict(zip(self.parm_names,scaling_factors))   
            
         
    def set_update(self,**set_update):
        #update settings:
        for key in set_update.keys():
            if type(set_update[key])==dict:
                for k in set_update[key]:
                    self.set[key][k]=set_update[key][k]
            else:
                self.set[key]=set_update[key]
                        
        self.funs = self.set['funs']
        self.parm_names = []
        self.p0 = []        
        self.p_lower = []
        self.p_upper = []
        self.deviation = 1
        
        for n,fun in enumerate(self.funs):
            for m,parm_name in enumerate(self.set['parms'][n]):
                if not parm_name in self.set['common_parms']:
                    self.parm_names.append('fun'+str(n+1)+'_'+parm_name)
                    self.p0.append(self.set['p0'][n][m])
                    self.p_lower.append(self.set['p_lower'][n][m])
                    self.p_upper.append(self.set['p_upper'][n][m])
                else:
                    if parm_name not in self.parm_names:
                        self.parm_names.append(parm_name)
                        self.p0.append(self.set['p0'][n][m])
                        self.p_lower.append(self.set['p_lower'][n][m])
                        self.p_upper.append(self.set['p_upper'][n][m])
        
        self.p0_dict = dict(zip(self.parm_names,self.p0))

    # =============================================================================
    # Actual Fit Function
    # =============================================================================
    def fit(self,**set_update):
        # =============================================================================
        # define chi2 function
        # ============================================================================= 
        def make_D(p0_dict):
            D = np.zeros((self.no_comps,len(self.x)))
            for n,fun in enumerate(self.funs):
                fun_name='fun'+str(n+1)+'_'
                fun_parms=[]
                for fun_parm in self.set['parms'][n]:
                    if fun_parm in self.set['common_parms']:
                        fun_parms.append(p0_dict[fun_parm]/self.scaling_factors[fun_parm])
                    else:
                        fun_parms.append(p0_dict[fun_name+fun_parm]/self.scaling_factors[fun_name+fun_parm])      
                D[n,:]=self.funs[n](self.x,*fun_parms)
            return D
            
        def make_DADS(Z,D):
            G = D@D.T
            if np.isclose(np.linalg.det(G),0):
                G = G+1e-12*np.eye(*G.shape)
                DADS=(Z@D.T)@(inv(G))
            else:
                DADS=(Z@D.T)@(inv(G))
            return DADS
    
        # =============================================================================
        #  Set initial values for fit:               
        # =============================================================================
        pfit_dict = self.p0_dict
        D = make_D(pfit_dict)
        if not self.set['fixed DADS']:
            DADS = make_DADS(self.Z,D)

        # =============================================================================
        #  Start Fit:
        # =============================================================================
        print('==================================\n'+'Starting Fit'+'\n==================================')  

        for iteration in range(self.set['iterations']):
            #---------------------
            #Start Iterations  
            print(f'Iteration {iteration}')
            
            def chi2(*parms): # takes parms as list of lists of parameters of functions       
                values = dict(zip(self.parm_names,parms)) 
                D = make_D(values)
                Z_fit=DADS@D
                self.deviation = np.nansum((self.Z-Z_fit)**2)
                
                return self.deviation

            # =============================================================================
            # create Minuit object and set fit settings            
            m=Mi(chi2, **pfit_dict, name=self.parm_names)
            # m=Mi(chi2, *p0_fit,name=self.parm_names)
            for n,p in enumerate(self.parm_names):
                m.limits[p]=(self.p_lower[n],self.p_upper[n])
            if self.set['errors']:
                m.errors=self.set['errors']
            if self.set['tol']:
                m.tol=self.set['tol']
            if self.set['strategy']:
                m.strategy=self.set['strategy']
            if self.set['print_level']:
                m.print_level=self.set['print_level']
            
            # =============================================================================
            # Run Fit
            if self.set['method']:
                m.scipy(self.set['method'])
            else:
                m.migrad()  

            # =============================================================================
            # Update Data for next iteration
            pfit_dict = dict(zip(self.parm_names,list(m.values)))
            D = make_D(pfit_dict)
            DADS = make_DADS(self.Z,D)
            print(self.deviation)
        
        #-------------------
        #End Iterations Loop. Unpack results.      
        #-------------------
        def unpack_parms(pfit_dict):
            p_fit = []
            for n,fun in enumerate(self.funs):
                fun_name='fun'+str(n+1)+'_'
                p_fit.append([])
                for fun_parm in self.set['parms'][n]:
                    if fun_parm in self.set['common_parms']:
                        p_fit[n].append(pfit_dict[fun_parm]/self.scaling_factors[fun_parm])
                    else:
                        p_fit[n].append(pfit_dict[fun_name+fun_parm]/self.scaling_factors[fun_name+fun_parm])
            return p_fit
        
        self.D = make_D(pfit_dict)
        self.DADS = make_DADS(self.Z,self.D)
        self.Z_fit = self.DADS@self.D
        self.p = unpack_parms(pfit_dict)
        self.p_dict = {parm: pfit_dict[parm]/self.scaling_factors[parm] for parm in pfit_dict.keys()}
        self.residuum = (self.Z_fit-self.Z)
        self.m = m
        
        # =============================================================================
        # print Results
        # =============================================================================      
        print(self.m.params)
        print(self.m.fmin)
        

        
    # =============================================================================
    #TODO: Singular Value Decomposition 
    # =============================================================================
    # class svd:
    #     def __init__(self,global_fit):
    #         self.u,self.s,self.v=np.linalg.svd(global_fit.Z)
            
    #     def plot(self):
    #         decays=[]
    #         specs=[]
    #         for n,i in enumerate(s):
    #             if i>(np.nanmax(s)/10):
    #                 decays.append(v[n,:]*i)
    #                 specs.append(u[:,n]*i)
            
    #         plot(fs[key]['fit'].wl,specs,title=key)
    #         timeplot(fs[key]['fit'].dt,decays,title=key,
    #                  x_break=5,labelpad=(0.1,0.1))
