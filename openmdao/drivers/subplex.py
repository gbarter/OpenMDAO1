from __future__ import print_function
import numpy as np
from variables import Variable    
from heuristic import Heuristic
from simplex import Simplex
import logging

class Subplex(Heuristic):
    def __init__(self, variables=None, xinit=None, model=None, options=None):
        Heuristic.__init__(self, variables, xinit, model, options) # don't use super because of multiple inheritance confusion later
        
        # Override Heuristic settings
        self.options['penalty'] = True
        self.npop = 1

        # Default values
        self.deltax    = np.zeros(len(xinit))
        self.psi       = 0.25
        self.omega     = 0.1
        self.nsmin     = 2
        self.nsmax     = 5

        # Default step size initialization for each variable
        self.step_size = np.zeros(self.nvar)
        for k in range(self.nvar):
            self.step_size[k] = self.variables[k].perturb(xinit[k], 5e-2) - xinit[k]

            
    def _initialize(self):
        Heuristic._initialize(self) # don't use super because of multiple inheritance confusion later
        self.x = [self.x[0]] # Should be xinit if not restarting

        
    def _get_next_subspace(self, dxabs):
        # Set subspace length restrictions
        n   = len(dxabs)
        ns0 = self.nsmin
        ns1 = n if n<self.nsmax else np.minimum(self.nsmax, n-self.nsmin)
        nsvec = range(ns0,ns1+1)
        
        # Assume input is already sorted by decreasing magnitude
        metric = [np.mean(dxabs[:k]) - np.mean(dxabs[k:]) for k in nsvec]

        # Return size of next subspace, catching initialization issues
        if np.all(metric == 0.0):
            return nsvec[-1]
        else:
            return nsvec[np.argmax(metric)]
        
            
    def _set_subspaces(self):
        # Reset subspaces
        self.subspaces = []
        
        # Sort changes in x by decreasing magnitude
        ix = np.flipud( np.argsort( np.abs(self.deltax) ) )
        dxabs = np.abs(self.deltax[ix])

        # Now partition into subspaces until all are used up
        while len(ix) > 0:
            ns = self._get_next_subspace(dxabs)
            self.subspaces.append( ix[:ns].tolist() )
            ix = ix[ns:]
            dxabs = dxabs[ns:]

        # Check for valid subspace partitioning
        badFlag = False
        nsubs = np.array([len(m) for m in self.subspaces])
        allsubs = [i for m in self.subspaces for i in m]
        if np.any(nsubs < self.nsmin):
            print('A subspace is too small!')
            badFlag = True
        elif np.any(nsubs > self.nsmax):
            print('A subspace is too big!')
            badFlag = True
        elif len(np.unique(allsubs)) != self.nvar:
            print('Missing a variable in the subspaces!',np.setdiff1d(np.unique(allsubs), np.arange(self.nvar)))
            badFlag = True
            
        if badFlag:
            print(self.subspaces)
            raise Exception('Subspace error')
        
        
    def _subspace_simplex(self):
        # Initialize options dictionary for each Simplex call
        ioptions = {'penalty':True,
                    'tol':-self.psi,
                    'penalty_multiplier':100,
                    'global_search':False,
                    'adaptive_simplex':False,
                    'generations':500,
                    'nstall':1000,
                    'restart':False}

        # Save current 
        xorig = np.array( self.x[0][:] )
        xnew  = np.array( self.x[0][:] )
        for isub in self.subspaces:
            # Variables for this subspace
            ivar   = [self.variables[m] for m in isub]
            ixinit = xnew[isub].tolist()
            
            # Set options for this subspace
            ioptions['population'] = len(isub)+1
            ioptions['step_size']  = self.step_size[isub]

            # Define calling function for this subspace
            def imodel( xin ):
                xsub = xnew.copy()
                xsub[isub] = xin
                obj, con, total = self._evaluate_input( [xsub.tolist()] )
                return (obj, con)
            
            # Execute simplex and store result
            isimplex = Simplex(ivar, ixinit, imodel, ioptions)
            isimplex.logger.setLevel(logging.CRITICAL)
            xsub, _, _ = isimplex.optimize()
            xnew[isub] = xsub

        # Compute difference in design variables so that we can set next steps and subspaces
        self.logger.setLevel(logging.INFO)
        self.deltax = xnew - xorig

        # Store new design variable vector
        self.x = [xnew.tolist()]
        

    def _set_stepsizes(self):
        # Set initial guesses of step scaling based on previous progress and number of subspaces
        if len(self.subspaces) > 1:
            scale = np.sum(np.abs(self.deltax)) / np.sum(np.abs(self.step_size))
        else:
            scale = self.psi

        # Bound the scaling and set the initial step size estimate
        scale = np.minimum( np.maximum(scale, self.omega), 1/self.omega)
        step  = scale * self.step_size

        # Fix orientation for initial simplexes
        ind = (self.deltax == 0.0)
        self.step_size = np.sign(self.deltax) * np.abs(step)
        self.step_size[ind] = -step[ind]


    def _termination_test(self):
        xarray = np.array( self.x[0] )
        option1 = np.abs(self.deltax / xarray)
        option2 = np.abs(self.step_size / xarray)
        self.terminateFlag = np.maximum(option1, option2).max() <= self.options['tol']


    def _iterate(self, kiter):
        self._set_subspaces()
        self._subspace_simplex()
        self._set_stepsizes()
        self._termination_test()

        # Store best designs
        self.xglobal     = self.x[0][:]
        #obj, con, total  = self._evaluate_input( [self.xglobal] )
        self.objGlobal   = self.obj[0]
        self.conGlobal   = self.con[0]
        self.totalGlobal = self.total[0]
