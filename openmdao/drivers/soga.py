from __future__ import print_function
import numpy as np
from variables import Variable    
from heuristic import Heuristic

class SOGA(Heuristic):
    def __init__(self, variables, xinit, model, options):
        super(SOGA, self).__init__(variables, xinit, model, options)
        
        # Added options
        if self.options['probability_of_mutation'] < 0.0:
            self.options['probability_of_mutation'] = 1.0 / float(self.nvar) # NSGA2 approach
        if not (self.options['population']%2 == 0): self.npop += 1

        # Offspring
        self.xchild     = None
        self.conChild   = np.inf * np.ones(self.npop)
        self.objChild   = np.inf * np.ones(self.npop)
        self.totalChild = np.inf * np.ones(self.npop)

    
    def _evaluate(self):
        super(SOGA, self)._evaluate()

        # Now add children
        if not (self.xchild is None):
            self.objChild, self.conChild, self.totalChild = self._evaluate_input(self.xchild)


    def _mating_options(self, p1, p2, obj1, obj2):
        if obj1 < obj2:
            return self.x[p1][:]
        elif obj2 < obj1:
            return self.x[p2][:]
        else:
            return (self.x[p1][:] if np.random.rand() < 0.5 else self.x[p2][:])

        
    def _mating_selection(self):
        # Constrained binary tournament selection from NSGA2
        # Use total objective + constraint penalty as merit function though
        self.xmate = self.x[:]
        
        # Random tournament pairing:
        tour1 = np.random.permutation(self.npop)
        tour2 = np.random.permutation(self.npop)

        for n in range(self.npop):
            p1   = tour1[n]
            p2   = tour2[n]
            obj1 = self.obj[p1]
            obj2 = self.obj[p2]
            con1 = self.con[p1]
            con2 = self.con[p2]
            tot1 = self.total[p1]
            tot2 = self.total[p2]

            # If using cumulative penalties
            if self.options['penalty']:
                self.xmate[n] = self._mating_options(p1, p2, tot1, tot2)
            else:
                # Satisfy constraints before objective
                if (con1<=0.0 and con2<=0.0): # both feasible
                    self.xmate[n] = self._mating_options(p1, p2, obj1, obj2)
                else:
                    self.xmate[n] = self._mating_options(p1, p2, con1, con2)

                    
    def _crossover(self):
        # Constants
        pcross = self.options['probability_of_crossover']
        eta_c  = self.options['crossover_index']
        
        # Every 2 parents have 2 children, so matrices are same size
        self.xchild = self.xmate[:]
        parents = np.random.permutation(self.npop)
        p1 = parents[0::2]
        p2 = parents[1::2]
        for n in range(len(p1)):
            # Initialize parents and child design variables
            x1 = self.xmate[p1[n]][:]
            x2 = self.xmate[p2[n]][:]
            y1 = x1[:]
            y2 = x2[:]

            # Do crossover variable by variables
            if np.random.rand() < pcross:
                for k in range(self.nvar):
                    y1[k], y2[k] = self.variables[k].cross(x1[k], x2[k], eta_c)
                        
            # Store child values
            self.xchild[p1[n]] = y1[:]
            self.xchild[p2[n]] = y2[:]

            
    def _mutation(self):
        # Constants
        pmutate = self.options['probability_of_mutation']
        eta_m   = self.options['mutation_index']
        
        inds = np.where( np.random.random((self.npop, self.nvar)) < pmutate)
        for n,k in zip(inds[0], inds[1]):
            self.xchild[n][k] = self.variables[k].mutate( self.xchild[n][k], eta_m )

            
    def _combine(self):
        # Combine all data
        self.con   = np.r_[self.con  , self.conChild]
        self.obj   = np.r_[self.obj  , self.objChild]
        self.total = np.r_[self.total, self.totalChild]
        self.x.extend( self.xchild )
        
        assert len(self.x) == 2*self.npop
        assert self.con.size == 2*self.npop
        assert self.obj.size == 2*self.npop
        assert self.total.size == 2*self.npop
        
        # Remove child matrices
        self.objChild   = None
        self.conChild   = None
        self.totalChild = None
        self.xchild     = None
        self.xmate      = None

        
    def _survival_selection(self):
        self._combine()
        self._rank()

        # If using penalty, then designs are already ranked as we want them
        if self.options['penalty']:
            nextgen = np.arange(self.npop, dtype=np.int64)
        else:
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
        self.total = self.total[nextgen]
        self.obj   = self.obj[nextgen]
        self.con   = self.con[nextgen]
        xobj       = self.x[:]
        self.x     = self.x[:self.npop]
        for n in range(self.npop):
            self.x[n] = xobj[nextgen[n]][:]

        # Store the best performers
        self.objGlobal   = self.obj[0]
        self.conGlobal   = self.con[0]
        self.totalGlobal = self.total[0]
        self.xglobal     = self.x[0][:]
            
    def _iterate(self, kiter):
        self._mating_selection()
        self._crossover()
        self._mutation()
        self._evaluate()
        self._survival_selection()
