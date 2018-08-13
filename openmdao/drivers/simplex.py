from __future__ import print_function
import numpy as np
from variables import Variable    
from heuristic import Heuristic
from scipy import optimize

class Simplex(Heuristic):
    def __init__(self, variables=None, xinit=None, model=None, options=None):
        Heuristic.__init__(self, variables, xinit, model, options) # don't use super because of multiple inheritance confusion later

        # Override Heuristic settings
        self.npop = 1
        #self.options['population'] = 1
        
        # Local class variables:
        # Index of continuous variables
        self.icont = None
        
        # Lower and upper bound arrays
        self.LB = None
        self.UB = None

        # Need to stick with numpy arrays here
        self.xinit = np.array( self.xinit )
        

    def _initialize(self):
        if self.options['restart']:
            self._load_restart()
            self.x = [self.x[0]]
        else:
            self.x = [self.xinit.tolist()]
            
        # Initialize self variables
        icont = []
        LB    = []
        UB    = []
        self.count = 0

        # Find continuous variables and their bounds
        for k in range(self.nvar):
            if self.variables[k].get_type() == type(0.0):
                icont.append( k )
                LB.append( self.variables[k].lower_bound )
                UB.append( self.variables[k].upper_bound )

        # Store class variables as numpy arrays
        self.LB    = np.array( LB )
        self.UB    = np.array( UB )
        self.icont = np.array( icont )

        # Return no-bounds versions of continuous variables
        xcont = np.array( self.x[0][:] )[self.icont]
        x0    = self._transformX( xcont )
        return x0
    
        
    def _transformX(self, x0):
        # Idea taken from http://www.mathworks.com/matlabcentral/fileexchange/8277-fminsearchbnd--fminsearchcon
        # And https://github.com/alexblaessle/constrNMPy/blob/master/constrNMPy/constrNMPy.py	
        x0u = np.maximum(self.LB, np.minimum(x0, self.UB))
        x0u = 2.0 * (x0u - self.LB) / (self.UB-self.LB) - 1.0
        #shift by 2*pi to avoid problems at zero in fmin otherwise, the initial simplex is vanishingly small
        x0u = 2.0*np.pi + np.arcsin( x0u )
        return x0u

        
    def _itransformX(self, x, eps=1e-12):
        # Idea taken from http://www.mathworks.com/matlabcentral/fileexchange/8277-fminsearchbnd--fminsearchcon
        # And https://github.com/alexblaessle/constrNMPy/blob/master/constrNMPy/constrNMPy.py	

        # Add offset if necessary to avoid singularities
        LB = self.LB.copy()
        LB[LB == 0.0] = eps
        
        # Do transformation
        xp = 0.5*(np.sin(x)+1.) * (self.UB - LB) + LB
        xp = np.maximum(self.LB, np.minimum(xp, self.UB))

        # Repack with integer and boolean variables that are being held static
        xout = self.xinit.copy() 
        xout[self.icont] = xp
        return xout	

        
    def _iterate(self):
        pass

    def _callback(self, xk):
        self.count += 1
        if self.count%100 == 0:
            self.x = [ self._itransformX( xk ).tolist() ]
            self._write_restart()          
            
    def _myrun(self, x):
        # Convert to bounded space that assembly expects
        xrun = self._itransformX(x)

        # Evaluate all and return penalty version
        obj, con, total = self._evaluate_input( [xrun.tolist()] )
        return total[0]

    
    def optimize(self):
        # Set bounds arrays and get initial starting array
        x0 = self._initialize()
        
        # Run Scipy Nelder-Mead simplex
        res = optimize.minimize(self._myrun, x0, method='Nelder-Mead',
                                tol=self.options['tol'], callback=self._callback,
                                options={'maxiter':self.options['generations'], 'disp':True})
        
        # Store Scipy ouput in unbounded space
        self.xglobal = self._itransformX( res.x ).tolist()
        self.objGlobal, self.conGlobal, _ = self._evaluate_input( [self.xglobal] )

        # Final logging
        self.x = self.xglobal[:]
        self._write_restart()
        
        return (self.xglobal, self.objGlobal[0], self.conGlobal[0])
