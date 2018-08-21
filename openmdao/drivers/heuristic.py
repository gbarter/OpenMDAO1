from __future__ import print_function
import cPickle as pickle        
import csv
import numpy as np
from variables import Variable    
import time
import logging
import sys
import os.path

RSTFILE = 'heuristic.restart'
LOGFILE = 'heuristic.log'
LOGNAME = 'mylogger'

class Heuristic(object):
    def __init__(self, variables=None, xinit=None, model=None, options=None):
        object.__init__(self) # Don't user "super" because of multiple inheritance issues later
        
        # Store list of variables and ensure type
        self.variables = variables
        self.nvar      = len(variables)
        assert isinstance(variables, list)
        assert isinstance(variables[0], Variable)

        # Store objective and constraint functions
        self.fmodel = model

        # Logging instance
        self.logger = logging.getLogger(LOGNAME)
        self.logger.setLevel(logging.DEBUG)
        
        # Save options (make sure population is even for pairing purposes)
        self.options = options
        self.npop    = self.options['population']

        # Initialize design variable matrix
        if isinstance(xinit, np.ndarray):
            self.xinit = xinit.tolist()
        elif isinstance(xinit, list):
            self.xinit = xinit[:]
        else:
            raise ValueError('Initial design variable vector must be a list or numpy array')
        self.x = [[None]*self.nvar for n in range(self.npop)]

        # Initialize constraint and performance vector
        self.con   = None
        self.obj   = None
        self.total = None

        # Containers to store the global best
        self.objGlobal   = None
        self.conGlobal   = None
        self.totalGlobal = None
        self.xglobal     = None

    def _generate_population(self, npop):
        
        if self.options['global_search']:
            # Populate design variables with Latin Hypercube initialization
            x = np.inf * np.ones((npop, self.nvar))
            for k in range(self.nvar):
                x[:,k] = self.variables[k].sample_lhc(npop)
                
        else:
            # Build simplex meant for neighborhood search (taken from scipy fmin)
            x = np.tile( self.xinit, (npop, 1))
            for k in range(self.nvar):
                x[k,k] = 1.05 * x[k,k] if x[k,k] != 0.0 else 2.5e-4

        # Make sure we're within bounds
        for k in range(self.nvar):
            x[:,k] = self.variables[k].bound( x[:,k] )
                
                
        return x.tolist()

    
    def _write_restart(self):
        with open(RSTFILE,'wb') as fp:
            pickle.dump(self.x, fp)

            
    def _load_restart(self):
        with open(RSTFILE,'rb') as fp:
            self.x = pickle.load(fp)

        # Handle improper number of population
        if len(self.x) > self.npop:
            self.x = self.x[:self.npop]
        elif len(self.x) < self.npop:
            nadd = self.npop - len(self.x)
            xadd = self._generate_population(nadd)
            self.x.extend( xadd )
        assert len(self.x) == self.npop

        # Handle improper number of variables: error here
        if len(self.x[0]) != self.nvar:
            raise ValueError('Inconsistent number of variables.  Restart file: '+str(len(self.x[0]))+'  Expected: '+str(self.nvar))
        

            
    def _initialize(self):
        if self.options['restart'] and os.path.exists(RSTFILE):
            self._load_restart()
        else:
            # Be sure initial state is included in population
            self.x = [self.xinit]
            self.x.extend( self._generate_population(self.npop-1) )

    
    def _evaluate_input(self, x):
        
        # Initialize outputs
        nx  = len(x)
        obj = np.inf * np.ones((nx,))
        con = np.inf * np.ones((nx,))

        for n in range(nx):
            obj[n], con[n] = self.fmodel( x[n] )

        # Determine penalty
        coeff = self.options['penalty_multiplier'] ** np.ceil( np.log10(np.abs(obj.min())) )
        total = obj + coeff*con
        
        return obj, con, total
    
    def _evaluate(self):
        self.obj, self.con, self.total = self._evaluate_input(self.x)

        
    def _rank(self):
        # Rank all designs first
        isort = np.argsort(self.total) if self.options['penalty'] else np.argsort(self.obj)
        self.total = self.total[isort]
        self.obj   = self.obj[isort]
        self.con   = self.con[isort]
        xobj       = self.x[:]
        for n in range(len(isort)):
            xobj[n] = self.x[isort[n]][:]
        self.x = xobj[:]
        return isort

    
    def _iterate(self):
        raise NotImplementedError()

    
    def optimize(self):
        # Setup
        self._initialize()
        self._evaluate()

        # Logging initialization
        self.logger.info('Iter\tTime\tObjective\t\tConstraint')
        
        # Iteration over generations
        ngen       = self.options['generations']
        tol        = self.options['tol']
        objHistory = []
        conHistory = []
        iteration  = 1
        nconverge  = self.options['nstall']
        nround     = int( np.round(np.abs(np.log10(tol))) )

        def convtest(x):
            return (np.round(np.sum(np.diff(x[-nconverge:])), nround) == 0.0)

        for k_iter in range(ngen):
            t = time.time()
        
            # Perform evolution for this generation
            self._iterate(k_iter)

            # Store in history
            objHistory.append( self.objGlobal )
            conHistory.append( self.conGlobal )
            
            # Logging
            self.logger.info(str(k_iter)+':\t'+
                             '{0:.3f}'.format(time.time() - t)+'\t'+
                             '{0:.6f}'.format(self.objGlobal)+'\t\t'+
                             '{0:.6f}'.format(self.conGlobal))

            if (k_iter%10 == 0): self._write_restart()

            # Check for convergence
            if ( (k_iter > nconverge) and
                 convtest(objHistory) and convtest(conHistory) ):
                break

            # Population based check
            xarray = np.array( self.x )
            if ( (np.max(np.abs(self.obj[1:] - self.obj[0])) <= tol) and
                 (np.max(np.abs(self.con[1:] - self.con[0])) <= tol) ):
                break
            

        # Final logging
        self._write_restart()
        
        return (self.xglobal, self.objGlobal, self.conGlobal)
