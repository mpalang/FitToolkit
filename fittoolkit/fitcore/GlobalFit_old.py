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


from fittoolkit.utils.tools import convolve,FitProgress,FitParamRegistry

default_settings = {                
                'common_parms': [], # Tuple of parm names that are common between different functions.
                'method': 'Powell', # 'migrad','L-BFGS-B','TNC','COBYLA','SLSQP','trust-constr','CG','Powell','BFGS',
                'scan': False,
                'simplex': False,
                'hesse': False,
                'tol':None,
                'errors':None,
                'strategy':None, # 
                'print_level':0,
                'iterations':7,
                'fixed DADS': False,
                'scale': False,
                'irf': None, #must be a function
                'irf_parms': None,
                'irf_p0': None,
                'irf_p_lower': None,
                'irf_p_upper': None,
                    }

class GlobalFit:
    """
    Creactes global fit Object to fit 2D spectroscopic Data and return DADS.
    Possible Fit Methods:
        'L-BFGS-B','TNC','COBYLA','SLSQP','trust-constr','CG','Powell','BFGS',
        #Additional Information required: 'trust-krylov','trust-ncg','dogleg','Newton-CG','trust-exact'
    
    callback: function that handles progres update in form of FitProgress object.
    
    """   
    # =============================================================================
    #     Initialize the global_fit object
    # =============================================================================  
    # x: np.ndarray
    # y: np.ndarray
    # Z: np.ndarray
    # p0: list
    # funs: list
    # settings: dict
    
    # p_lower: list = []
    # p_upper: list = []
    
    # irf = None
    
    def __init__(self,x,y,Z,name="Data",settings=None,callback=None):
        self.callback = callback
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
        Default settings.

        """
        self.set = default_settings
        self.set={
                'funs':None, # is a list of functions.
                'parms':[], # is a list of tuples with parm names
                'p0':[], #is a list of tuples of guesses
                'p_lower':[], #is a list of tuples of lower boundaries. Can be None.
                'p_upper':[], #is a list of tuples of upper boundaries. Can be None.
                
                'common_parms': [], # Tuple of parm names that are common between different functions.
                'method': 'Powell', # 'migrad','L-BFGS-B','TNC','COBYLA','SLSQP','trust-constr','CG','Powell','BFGS',
                'scan': False,
                'simplex': False,
                'hesse': False,
                'tol':None,
                'errors':None,
                'strategy':None, # 
                'print_level':0,
                'iterations':7,
                'fixed DADS': False,
                'scale': False,

                'irf': None, #must be a function
                'irf_parms': None,
                'irf_p0': None,
                'irf_p_lower': None,
                'irf_p_upper': None,
            }   

                    
    def pack_fit_parms_new(self):#TODO finish this alteration!
      p_indices = []
      params = FitParamRegistry()
      for n,fun in enumerate(self.set['funs']):
          fun_name = 'fun'+str(n+1)+'_'
          p_indices.append([])
          for m,name in enumerate(self.set['parms'][n]):
              is_common = (name in self.set['common_parms'])
              if not is_common:
                  name = fun_name+name
              params.add(name,self.set['p0'][n][m],
                             self.set['p_lower'][n][m],
                             self.set['p_upper'][n][m],
                             is_common)

              
      self.p_indices = p_indices
      self.params = params


    def pack_fit_parms(self):
        fit_p0 = []
        fit_parms = []
        fit_pl = []
        fit_pu = []
        fit_p_indices = []
        cp_index={}
        i = 0
        for n,fun in enumerate(self.set['funs']):
            fun_name = 'fun'+str(n+1)+'_'
            fit_p_indices.append([])
            for m,name in enumerate(self.set['parms'][n]):
                if name in self.set['common_parms']:
                    if name in fit_parms:
                        fit_p_indices[n].append(cp_index[name])
                        continue
                    fit_parms.append(name)
                    fit_p0.append(self.set['p0'][n][m])
                    fit_pl.append(self.set['p_lower'][n][m])
                    fit_pu.append(self.set['p_upper'][n][m])
                    cp_index[name]=i
                    fit_p_indices[n].append(i)
                else:
                    fit_p0.append(self.set['p0'][n][m])
                    fit_parms.append(fun_name+name)
                    fit_pl.append(self.set['p_lower'][n][m])
                    fit_pu.append(self.set['p_upper'][n][m])
                    fit_p_indices[n].append(i)
                i+=1
        
        if self.set['irf']:
            fun_name = 'irf_'
            irf_p_indices = []
            for m,name in enumerate(self.set['irf_parms']):
                if name in self.set['common_parms']:
                    if name in fit_parms:
                        irf_p_indices.append(cp_index[name])
                        continue
                    fit_parms.append(name)
                    fit_p0.append(self.set['irf_p0'][m])
                    fit_pl.append(self.set['irf_p_lower'][m])
                    fit_pu.append(self.set['irf_p_upper'][m])
                    cp_index[name]=i
                    irf_p_indices.append(i)
                else:
                    fit_p0.append(self.set['irf_p0'][m])
                    fit_parms.append(fun_name+name)
                    fit_pl.append(self.set['irf_p_lower'][m])
                    fit_pu.append(self.set['irf_p_upper'][m]) 
                    irf_p_indices.append(i)
                i+=1
        
        self.parm_names = tuple(fit_parms)
        self.p0 = tuple(fit_p0)
        self.p_lower = tuple(fit_pl)
        self.p_upper = tuple(fit_pu)
        self.fun_indices = tuple(fit_p_indices)
        if self.set['irf']:
            self.irf_indices = tuple(irf_p_indices)


    def set_update(self,**set_update):
        #update settings:
        for key in set_update.keys():
            if type(set_update[key])==dict:
                for k in set_update[key]:
                    self.set[key][k]=set_update[key][k]
            else:
                self.set[key]=set_update[key]
                        
        self.funs = self.set['funs']
        self.pack_fit_parms()
        self.deviation = 1 #this is just to have a value for deviation before the fit is run. It will be overwritten by the fit function.
    
    
    def check_data(self):
        if self.Z.shape != (np.squeeze(self.y.shape),np.squeeze(self.x.shape)):
            raise ValueError()
        
        # bring parameters to same magnitude for better fit:
        self.scaling_factors=[1]*len(self.p0)

        
        if self.set['scale']:
            self.p0 = list(self.p0)
            self.p_lower = list(self.p_lower)
            self.p_upper = list(self.p_upper)
            for n,parm in enumerate(self.p0):
                factor = 10**(-np.floor(np.log10(np.abs(parm))))
                if factor and not np.isinf(factor):
                    self.scaling_factors[n] = factor
                    self.p0[n] = self.p0[n]*self.scaling_factors[n]
                    self.p_lower[n] = self.p_lower[n]*self.scaling_factors[n]
                    self.p_upper[n] = self.p_upper[n]*self.scaling_factors[n]
        
            self.scaling_factors = tuple(self.scaling_factors)
        
            self.p0 = tuple(self.p0)
            self.p_lower = tuple(self.p_lower)
            self.p_upper = tuple(self.p_upper)
        

    # =============================================================================
    # Actual Fit Function
    # =============================================================================
    def fit(self,**set_update):
        # =============================================================================
        # define local functions:
        # ============================================================================= 
        def getelements(L,indices):
            return [l for n,l in enumerate(L) if n in indices]
        
        def make_D(*p):
            D = np.zeros((self.no_comps,len(self.x)))
            for n,fun in enumerate(self.funs):  
                fun_parms = (np.array(getelements(p,self.fun_indices[n]))/
                            np.array(getelements(self.scaling_factors,self.fun_indices[n])))
                
                if self.set['irf']:
                    irf_parms = (np.array(getelements(p,self.irf_indices))/
                            np.array(getelements(self.scaling_factors,self.irf_indices)))
                    D[n,:]=convolve(self.x,self.funs[n](self.x,*fun_parms),
                                    lambda x: self.set['irf'](x,*irf_parms))
                else:
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
        fit_dict = dict(zip(self.parm_names,self.p0))
        D = make_D(*self.p0)
        if self.set['fixed DADS']:
            DADS = self.set['fixed DADS']
        else:
            DADS = make_DADS(self.Z,D)

        def chi2(*p):
            D = make_D(*p)
            Zfit = DADS@D
            deviation = np.nansum((self.Z-Zfit)**4) # why not use chi4 instead of chi2??
            return deviation 
        
        # =============================================================================
        # create Minuit object and set fit settings            
        m=Mi(chi2, **fit_dict, name=self.parm_names)
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
            
        if self.set['simplex']:
            if self.set['print_level']>-1:
                if self.callback:
                    self.callback('running simplex')
            m.simplex()
            
        # =============================================================================
        #  Start Fit:
        # =============================================================================
        if self.callback:
            self.callback('start fit')
        
        self.deviation = [] # create list so you can follow the error throughout the iterations.
        for iteration in range(self.set['iterations']):
            #---------------------
            #Start Iterations  
            if self.set['print_level']>-1:
                print(f'Iteration {iteration+1}')
            
            # =============================================================================
            # Run Fit
            if self.set['method']=='migrad':
                # print('running migrad')
                m.migrad() 
            elif self.set['method']:
                # print(f"running {self.set['method']}")
                m.scipy(self.set['method'])
 
            if self.set['hesse']:
                # print('calculating hesse')
                m.hesse()

            # =============================================================================
            # Update Data for next iteration
            fit_dict = dict(zip(self.parm_names,list(m.values)))
            D = make_D(*m.values)
            if self.set['fixed DADS']:
                DADS = self.set['fixed DADS']
            else:
                DADS = make_DADS(self.Z,D)
            deviation = chi2(*m.values)
            self.deviation.append(deviation)
            if self.set['print_level']>0:
                print(f'Chi2: {deviation}')
            if self.callback:
                fp = FitProgress(iteration=iteration,total_iterations=self.set['iterations'],
                            chi2=deviation,parameters=m.values)
                self.callback(fp)
        
        #-------------------
        #End Iterations Loop. Unpack results.      
        #-------------------
        fit_dict = dict(zip(self.parm_names,
                            np.array(m.values)/np.array(self.scaling_factors)))
        
        # pack fit results
        p_fit = []
        for n,fun in enumerate(self.funs):
            fun_name='fun'+str(n+1)+'_'
            p_fit.append([])
            for fun_parm in self.set['parms'][n]:
                if fun_parm in self.set['common_parms']:
                    p_fit[n].append(fit_dict[fun_parm])
                else:
                    p_fit[n].append(fit_dict[fun_name+fun_parm])
        
        self.D = make_D(*m.values)
        self.DADS = make_DADS(self.Z,self.D)
        self.Z_fit = self.DADS@self.D
        self.p = list(p_fit)
        self.p_dict = fit_dict
        self.residuum = (self.Z_fit-self.Z)
        self.m = m
        
        # =============================================================================
        # print Results
        # =============================================================================      
        if self.set['print_level']>0:
            print(m.params)
            print(m.fmin)
            if self.callback:
                self.callback(f'{m.params}')
                self.callback(f'{m.fmin}')
    

        
    # =============================================================================
    # TODO: Singular Value Decomposition 
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
