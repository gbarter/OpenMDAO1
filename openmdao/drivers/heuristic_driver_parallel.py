"""
OpenMDAO Sinlge Object Genetive Algorithm driver with multiprocessing support
"""

from __future__ import print_function

import numpy as np

# SOGA Additions
from heuristic_driver import HeuristicDriver
from heuristic import Heuristic, LOGNAME
from soga import SOGA
from sopso import SOPSO
import logging
import multiprocessing as mp


class HeuristicDriverParallel(HeuristicDriver, SOGA, SOPSO):
    """ Driver wrapper for the in-house single objective genetic algorithm (SOGA), 
    based on a matlab implementation of NSGA2.  Unique to this optimizer is the support 
    of continuous, discrete, and binary (boolean) design variables, in addition to 
    inequality and equality constraints.

    Options
    -------
    options['disp'] :  bool(True)
        Set to False to prevent printing of Scipy convergence messages
    options['population'] : int(200)
        Number of designs to carry through each generation
    options['generations'] : int(200)
        Number of generations to evolve each design
    options['probability_of_crossover'] : float(0.9)
        Probability of mating between two individuals

    """

    def __init__(self):
        HeuristicDriver.__init__(self)

        # Logging instance
        self.logger = logging.getLogger(LOGNAME)
        self.logger.setLevel(logging.DEBUG)


        # Heuristic additions
        self.npop       = None
        self.nvar       = None
        self.x          = None
        self.xinit      = None
        self.xglobal    = None
        self.con        = None
        self.obj        = None
        self.total      = None
        self.conGlobal   = None
        self.objGlobal   = None
        self.totalGlobal = None

        # SOGA additions
        self.xmate      = None
        self.xchild     = None
        self.conChild   = None
        self.objChild   = None
        self.totalChild = None

        # SOPSO additions
        self.xlocal     = None
        self.conLocal   = None
        self.objLocal   = None
        self.totalLocal = None
        self.velocity   = None
        
        
    def __call__(self, c, mydict):
        # Added hack for serialization of 'self' in multiprocessing thanks to StackOverflow
        obj, con = self._model(c)
        mydict[c] = (obj,con)
        #print(c, obj, con)
        #return (c, obj, con)

        
    def run(self, problem):
        # Same prep as parent class
        self.xinit = self._prerun(problem)

        # Initialize design vector list of lists
        self.x = [[None]*self.nvar for n in range(self.npop)]
        
        # SOGA additions
        self.xmate      = None
        self.xchild     = None
        self.conChild   = np.inf * np.ones(self.npop)
        self.objChild   = np.inf * np.ones(self.npop)
        self.totalChild = np.inf * np.ones(self.npop)
        
        # SOPSO additions
        self.xlocal     = [[None]*self.nvar for n in range(self.npop)]
        self.conLocal   = np.inf * np.ones(self.npop)
        self.objLocal   = np.inf * np.ones(self.npop)
        self.totalLocal = np.inf * np.ones(self.npop)
        self.velocity   = np.zeros( (self.npop, self.nvar) )
        
        # Optimize
        if self.options['optimizer'].lower() == 'soga':
            xresult, fmin, fcon = SOGA.optimize(self)

        elif self.options['optimizer'].lower() == 'sopso':
            xresult, fmin, fcon = SOPSO.optimize(self)

        else:
            raise ValueError('Unknown optimizer: '+self.options['optimizer']+
                            '.  Valid options are SOGA or SOPSO')

        # Store results locally
        fmin,fcon = self._model(xresult)

        # Process results (same as parent class)
        self._postrun(xresult, fmin, fcon)

                
    
    def _evaluate_input(self, x):
        '''
        This implementation of parallelization with the multiprocessing module is sub-optimal, 
        compared to the use of Pools and the Pool.map function.  There are a few reasons for this 
        that were discovered over arduous trial and error.  First, parallelization in Python within 
        a class is difficult.  This is because the multiprocessing module uses cPickle under the 
        hood to serialize data as it slings things from one process to another.  cPickle cannot do 
        'instance' methods well at all.  This is the hack of using the __call__ function found on 
        StackOverflow.  Second, becuase OpenMDAO ultimately updates 'self' when given a new design 
        vector instead of passing arguments to a function, we cannot use map functions easily because 
        serialization of 'self' essentially freezes a snapshot of 'self'.  When using the map functions, 
        every return value was the one from default values, when 'self' was frozen.  So, we have to use 
        Process, which kicks off the execution as soon as self is updated.  This also means worse process 
        management than what a pool would offer though.
        '''
        # Initialize outputs
        obj = np.inf * np.ones((self.npop,))
        con = np.inf * np.ones((self.npop,))

        # Set-up parallel execution
        manager = mp.Manager()
        results = manager.dict()
        #num_procs = 3 #mp.cpu_count()

        temp = []
        for c in range(self.npop):
            self._unpack( x[c] )
            p = mp.Process(target=self, args=(c,results))
            temp.append( p )
            p.start()
        for p in temp: p.join()
        for n in range(self.npop):
            try:
                obj[n] = results[n][0]
                con[n] = results[n][1]
            except KeyError: continue

        # Determine penalty
        coeff = 10.0 ** np.ceil( np.log10(np.abs(obj.min())) )
        total = obj + coeff*con
        
        return obj, con, total


    
    def _evaluate(self):
        if self.options['optimizer'].lower() == 'soga':
            SOGA._evaluate(self)
        else:
            SOPSO._evaluate(self)

            
    def _iterate(self, k_iter):
        if self.options['optimizer'].lower() == 'soga':
            SOGA._iterate(self, k_iter)
        else:
            SOPSO._iterate(self, k_iter)

    
