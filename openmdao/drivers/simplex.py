from __future__ import print_function
import numpy as np
from variables import Variable    
from heuristic import Heuristic

class Simplex(Heuristic):
    def __init__(self, variables=None, xinit=None, model=None, options=None):
        Heuristic.__init__(self, variables, xinit, model, options) # don't use super because of multiple inheritance confusion later

        # Size check
        assert self.npop == (self.nvar+1), 'Simplex size initialization error'
        
        # Override Heuristic settings
        self.options['penalty'] = True

        # Simplex adjustment constants
        self.rho   =  1.0
        if self.options['adaptive_simplex']:
            self.chi   = 1.0  + 2.0/self.nvar
            self.psi   = 0.75 - 0.5/self.nvar
            self.sigma = 1.0  - 1.0/self.nvar
        else:
            self.chi   =  2.0
            self.psi   =  0.5
            self.sigma =  0.5


    
    def _generate_point(self, coeff):
        xnew = [None] * self.nvar
        for k in range(self.nvar):
            xnew[k] = self.variables[k].simplex_point(self.xcentroid[k], self.x[-1][k], coeff)
        _, _, f_new = self._evaluate_input( [xnew] )
        return xnew, f_new[0]
    

    def _shrink_simplex(self):
        xarray = np.array( self.x )
        xdiff  = xarray - xarray[0,np.newaxis,:]
        xarray[1:,:] = xarray[0,np.newaxis,:] + self.sigma * xdiff[1:,:]
        self.x = xarray.tolist()
        self._evaluate()

        
    def _update_simplex(self):
        # Take centroid of all points up until last one
        xarray         = np.array( self.x )
        self.xcentroid = xarray[:-1,:].mean(axis=0)
        
        # Reflection point
        xreflect, f_r = self._generate_point(self.rho)

        # Initialize simplex shrink trigger
        shrinkFlag = False

        # Simplex logic
        if f_r < self.total[0]:
            # Check for expansion
            xexpand, f_e = self._generate_point(self.chi)
            if f_e < f_r:
                self.x[-1]     = xexpand
                self.total[-1] = f_e
            else:
                self.x[-1]     = xreflect
                self.total[-1] = f_r
            
        elif f_r < self.total[-2]:
            self.x[-1]     = xreflect
            self.total[-1] = f_r
            
        else:
            if f_r < self.total[-1]:
                # Outside contraction
                xcont, f_c = self._generate_point(self.psi)
                if f_c <= f_r:
                    self.x[-1]     = xcont
                    self.total[-1] = f_c
                else:
                    shrinkFlag = True

            else:
                # Inside contraction
                xcont, f_c = self._generate_point(-self.psi)
                if f_c < self.total[-1]:
                    self.x[-1]     = xcont
                    self.total[-1] = f_c
                else:
                    shrinkFlag = True

        # Now shrink the simplex
        if shrinkFlag:
            self._shrink_simplex()

        
    def _iterate(self, kiter):
        self._rank()
        self._update_simplex()

        # Store best designs
        self.xglobal     = self.x[0][:]
        obj, con, total  = self._evaluate_input( [self.xglobal] )
        self.objGlobal   = obj[0]
        self.conGlobal   = con[0]
        self.totalGlobal = total[0]
        
        
