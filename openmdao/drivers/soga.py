from __future__ import print_function
import numpy as np
from variables import Variable    
        
        
class SOGA:
    def __init__(self, variables, objFun, conFun, population=200, maxgen=200, probCross=0.9):

        # Store list of variables and ensure type
        self.variables = variables
        self.nvar      = len(variables)
        assert isinstance(variables, list)
        assert isinstance(variables[0], Variable)

        # Store objective and constraint functions
        self.fobj = objFun
        self.fcon = conFun

        # Save options (make sure population is even for pairing purposes)
        self.ngen = maxgen
        self.npop = population
        if not (population%2 == 0): self.npop += 1
        self.pcross  = probCross
        self.pmutate = 1.0 / float(self.nvar) # NSGA2 approach

        # Initialize design variable matrix
        self.x = None

        # Tournament victors for mating
        self.xmate = None

        # Offspring
        self.xchild = None

        # Initialize constraint and performance vector
        self.con = None
        self.obj = None
        
        self.conChild = None
        self.objChild = None
        
        
    def _initialize(self):
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

        if not (self.xchild is None):
            self.objChild = np.inf * np.ones((self.npop,))
            self.conChild = np.inf * np.ones((self.npop,))
            for n in xrange(self.npop):
                self.objChild[n] = self.fobj( self.xchild[n] )
                self.conChild[n] = self.fcon( self.xchild[n] )

        
    def _mating_selection(self):
        # Constrained binary tournament selection from NSGA2
        self.xmate = self.x[:]
        
        # Random tournament pairing:
        tour1 = np.random.permutation(self.npop)
        tour2 = np.random.permutation(self.npop)

        for n in xrange(self.npop):
            p1   = tour1[n]
            p2   = tour2[n]
            obj1 = self.obj[p1]
            obj2 = self.obj[p2]
            con1 = self.con[p1]
            con2 = self.con[p2]

            if (con1<=0.0 and con2<=0.0): # both feasible
                if obj1 < obj2:
                    self.xmate[n] = self.x[p1][:]
                elif obj2 < obj1:
                    self.xmate[n] = self.x[p2][:]
                else:
                    self.xmate[n] = self.x[p1][:] if np.random.rand() < 0.5 else self.x[p2][:]
            else:
                if con1 < con2:
                    self.xmate[n] = self.x[p1][:]
                elif con2 < con1:
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
        self.con = np.r_[self.con, self.conChild]
        self.obj = np.r_[self.obj, self.objChild]
        for n in xrange(self.npop):
            self.x.append( self.xchild[n] )

        # Remove child matrices
        self.objChild = None
        self.conChild = None
        self.xchild   = None
        self.xmate    = None

    def _rank(self):
        # Rank all designs first
        iobj     = np.argsort(self.obj)
        self.obj = self.obj[iobj]
        self.con = self.con[iobj]
        xobj     = self.x[:]
        for n in xrange(len(iobj)):
            xobj[n] = self.x[iobj[n]][:]
        self.x = xobj[:]
            
    def _survival_selection(self):
        self._combine()
        self._rank()
        
        # Cull out feasible designs first
        feasible  = np.nonzero(self.con <= 0.0)[0]
        nfeasible = len(feasible)
        if nfeasible >= self.npop:
            nextgen = feasible[:self.npop]
        else:
            # Append some infeasible designs
            infeasible = np.setdiff1d(np.arange(len(self.con)), feasible)
            # Sort these infeasible designs by closest to feasibility
            infeasible = infeasible[ np.argsort(self.con[infeasible]) ]
            
            nextgen = np.r_[feasible, infeasible]
            nextgen = nextgen[:self.npop]

        # Store data for next generation
        self.obj = self.obj[nextgen]
        self.con = self.con[nextgen]
        xobj     = self.x[:]
        self.x   = self.x[:self.npop]
        for n in xrange(self.npop):
            self.x[n] = xobj[nextgen[n]][:]
        

        
    def run(self):
        self._initialize()
        self._evaluate()

        iteration = 1
        while iteration < self.ngen:
            self._mating_selection()
            self._crossover()
            self._mutation()
            self._evaluate()
            self._survival_selection()
            iteration += 1
            #print(iteration, self.x[0], self.obj[0], self.con[0])
        return (self.x[0], self.obj[0], self.con[0])
