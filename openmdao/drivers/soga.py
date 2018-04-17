from __future__ import print_function
import cPickle as pickle        
import csv
import numpy as np
from variables import Variable    

RSTFILE  = 'soga.restart'
HISTFILE = 'soga.history'
OUTFILE  = 'soga.out'

class SOGA:
    def __init__(self, variables, objFun, conFun, population=200, maxgen=200, probCross=0.9, restart=False):

        # Store list of variables and ensure type
        self.variables = variables
        self.nvar      = len(variables)
        assert isinstance(variables, list)
        assert isinstance(variables[0], Variable)

        # Store objective and constraint functions
        self.fobj = objFun
        self.fcon = conFun

        # Save options (make sure population is even for pairing purposes)
        self.restart = restart
        self.ngen    = maxgen
        self.npop    = population
        self.pcross  = probCross
        self.pmutate = 1.0 / float(self.nvar) # NSGA2 approach
        if not (population%2 == 0): self.npop += 1

        # Initialize design variable matrix
        self.x = None

        # Tournament victors for mating
        self.xmate = None

        # Offspring
        self.xchild = None

        # Initialize constraint and performance vector
        self.con = None
        self.obj = None
        self.total = None
        
        self.conChild = None
        self.objChild = None
        self.totalChild = None
        

    def _write_restart(self):
        with open(RSTFILE,'wb') as fp:
            pickle.dump(self.x, fp)
            
    def _load_restart(self):
        with open(RSTFILE,'rb') as fp:
            self.x = pickle.load(fp)

    def _write_output(self):
        with open(OUTFILE, 'wb') as fcsv:
            writer = csv.writer(fcsv, delimiter=',')
            writer.writerow(['Popultion', self.npop])
            writer.writerow(['Variables', self.nvar])
            for n in xrange(self.npop):
                writer.writerow(self.x[n])
            writer.writerow(['Objective'])
            writer.writerow(self.obj.tolist())
            writer.writerow(['Constraints'])
            writer.writerow(self.con.tolist())

            
    def _initialize(self):
        if self.restart:
            self._load_restart()
        else:
            # Create design variable structure
            self.x = []
            for n in xrange(self.npop):
                self.x.append( [None for m in xrange(self.nvar)] )

            # Populate design variables
            for k in xrange(self.nvar):
                vals = self.variables[k].sample_lhc(self.npop)

                for n in xrange(self.npop):
                    self.x[n][k] = vals[n]
                
            
    def _evaluate(self):
        # Evaluate all designs
        self.obj = np.inf * np.ones((self.npop,))
        self.con = np.inf * np.ones((self.npop,))
        for n in xrange(self.npop):
            self.obj[n] = self.fobj( self.x[n] )
            self.con[n] = self.fcon( self.x[n] )

        # Determine penalty
        coeff = 10.0 ** np.ceil( np.log10(np.abs(self.obj.min())) )
        self.total = self.obj + coeff*self.con
        
        if not (self.xchild is None):
            self.objChild = np.inf * np.ones((self.npop,))
            self.conChild = np.inf * np.ones((self.npop,))
            for n in xrange(self.npop):
                self.objChild[n] = self.fobj( self.xchild[n] )
                self.conChild[n] = self.fcon( self.xchild[n] )
                
            self.totalChild = self.objChild + coeff*self.conChild

        
    def _mating_selection(self):
        # Constrained binary tournament selection from NSGA2
        # Use total objective + constraint penalty as merit function though
        self.xmate = self.x[:]
        
        # Random tournament pairing:
        tour1 = np.random.permutation(self.npop)
        tour2 = np.random.permutation(self.npop)

        for n in xrange(self.npop):
            p1   = tour1[n]
            p2   = tour2[n]
            tot1 = self.total[p1]
            tot2 = self.total[p2]
            if tot1 < tot2:
                self.xmate[n] = self.x[p1][:]
            elif tot2 < tot1:
                self.xmate[n] = self.x[p2][:]
            else:
                self.xmate[n] = self.x[p1][:] if np.random.rand() < 0.5 else self.x[p2][:]

                    
    def _crossover(self):
        # Every 2 parents have 2 children, so matrices are same size
        self.xchild = self.xmate[:]
        parents = np.random.permutation(self.npop)
        p1 = parents[0::2]
        p2 = parents[1::2]
        for n in xrange(len(p1)):
            # Initialize parents and child design variables
            x1 = self.xmate[p1[n]][:]
            x2 = self.xmate[p2[n]][:]
            y1 = x1[:]
            y2 = x2[:]

            # Do crossover variable by variables
            if np.random.rand() < self.pcross:
                for k in xrange(self.nvar):
                    y1[k], y2[k] = self.variables[k].cross(x1[k], x2[k])
                        
            # Store child values
            self.xchild[p1[n]] = y1[:]
            self.xchild[p2[n]] = y2[:]

            
    def _mutation(self):
        inds = np.where( np.random.random((self.npop, self.nvar)) < self.pmutate)
        for n,k in zip(inds[0], inds[1]):
            self.xchild[n][k] = self.variables[k].mutate( self.xchild[n][k] )

            
    def _combine(self):
        # Combine all data
        self.con   = np.r_[self.con  , self.conChild]
        self.obj   = np.r_[self.obj  , self.objChild]
        self.total = np.r_[self.total, self.totalChild]
        for n in xrange(self.npop):
            self.x.append( self.xchild[n] )

        # Remove child matrices
        self.objChild   = None
        self.conChild   = None
        self.totalChild = None
        self.xchild     = None
        self.xmate      = None

        
    def _rank(self):
        # Rank all designs first
        itot       = np.argsort(self.total)
        self.total = self.total[itot]
        self.obj   = self.obj[itot]
        self.con   = self.con[itot]
        xtot       = self.x[:]
        for n in xrange(len(itot)):
            xtot[n] = self.x[itot[n]][:]
        self.x = xtot[:]

        
    def _survival_selection(self):
        self._combine()
        self._rank()

        # Store data for next generation
        self.total = self.total[:self.npop]
        self.obj   = self.obj[:self.npop]
        self.con   = self.con[:self.npop]
        self.x     = self.x[:self.npop]

        
    def run(self):
        # Setup
        self._initialize()
        self._evaluate()

        # Logging initialization
        fhist = open(HISTFILE, 'w')
        fhist.write('Iter\tObjective\tConstraint\n')
        
        # Iteration over generations
        objHistory = np.empty((self.ngen,))
        conHistory = np.empty((self.ngen,))
        iteration  = 1
        while iteration <= self.ngen:
            # Perform evolution for this generation
            self._mating_selection()
            self._crossover()
            self._mutation()
            self._evaluate()
            self._survival_selection()

            # Store and log data
            objHistory[iteration-1] = self.obj[0]
            conHistory[iteration-1] = self.con[0]
            fhist.write(str(iteration)+':\t'+
                        '{0:.6f}'.format(self.obj[0])+'\t'+
                        '{0:.6f}'.format(self.con[0])+'\n')
            fhist.flush()
            if (iteration%10 == 0): self._write_restart()
            iteration += 1

        # Final logging
        fhist.close()
        self._write_restart()
        #self._write_output()
        
        return (self.x[0], self.obj[0], self.con[0])
