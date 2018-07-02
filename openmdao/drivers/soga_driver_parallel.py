"""
OpenMDAO Sinlge Object Genetive Algorithm driver with multiprocessing support
"""

from __future__ import print_function

import numpy as np

# SOGA Additions
from soga import SOGA
from soga_driver import SOGADriver
import multiprocessing as mp


class SOGADriverParallel(SOGADriver, SOGA):
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
        super(SOGADriverParallel, self).__init__()

        # SOGA additions
        self.npop       = None
        self.pmutate    = None
        self.xinit      = None
        self.x          = None
        self.xmate      = None
        self.xchild     = None
        self.con        = None
        self.obj        = None
        self.total      = None
        self.conChild   = None
        self.objChild   = None
        self.totalChild = None
        
        
    def __call__(self, c, mydict):
        # Added hack for serialization of 'self' in multiprocessing thanks to StackOverflow
        obj, con = self._model(c)
        mydict[c] = (obj,con)
        #print(c, obj, con)
        #return (c, obj, con)

        
    def run(self, problem):
        # Same prep as parent class
        self.xinit = self._prerun(problem)
                
        # SOGA specific prep
        self.pmutate = 1.0 / float(self.nvar) # NSGA2 approach
        self.npop    = self.options['population']
        if not (self.npop%2 == 0): self.npop += 1

        # Initialize design vector list of lists
        self.x = []
        for n in xrange(self.npop):
            self.x.append( [None for m in xrange(self.nvar)] )

        # Optimize
        xresult, fmin, fcon = self.optimize()

        # Store results locally
        fmin,fcon = self._model(xresult)

        # Process results (same as parent class)
        self._postrun(xresult, fmin, fcon)

                

    def _evaluate(self):
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
        self.obj = np.inf * np.ones((self.npop,))
        self.con = np.inf * np.ones((self.npop,))
        self.objChild = np.inf * np.ones((self.npop,))
        self.conChild = np.inf * np.ones((self.npop,))

        # Set-up parallel execution
        manager = mp.Manager()
        results = manager.dict()
        #num_procs = 3 #mp.cpu_count()

        temp = []
        for c in xrange(self.npop):
            self._unpack(self.x[c])
            p = mp.Process(target=self, args=(c,results))
            temp.append( p )
            p.start()
        for p in temp: p.join()
        for n in xrange(self.npop):
            try:
                self.obj[n] = results[n][0]
                self.con[n] = results[n][1]
            except KeyError: continue

        # Determine penalty
        coeff = 10.0 ** np.ceil( np.log10(np.abs(self.obj.min())) )
        self.total = self.obj + coeff*self.con
        
        if not (self.xchild is None):
            results = manager.dict()
            temp = []
            for c in xrange(self.npop):
                self._unpack(self.xchild[c])
                p = mp.Process(target=self, args=(c,results))
                temp.append( p )
                p.start()
            for p in temp: p.join()
            for n in xrange(self.npop):
                try:
                    self.objChild[n] = results[n][0]
                    self.conChild[n] = results[n][1]
                except KeyError: continue
                
        self.totalChild = self.objChild + coeff*self.conChild

