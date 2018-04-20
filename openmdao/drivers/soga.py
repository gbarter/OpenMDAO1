from __future__ import print_function
import cPickle as pickle        
import csv
import numpy as np
from variables import Variable    
import time

RSTFILE  = 'soga.restart'
HISTFILE = 'soga.history'
OUTFILE  = 'soga.out'

class SOGA:
    def __init__(self, variables, xinit, model,
                 population=200, maxgen=200, probCross=0.9,
                 restart=False, penalty=True, tol=1e-6):

        # Store list of variables and ensure type
        self.variables = variables
        self.nvar      = len(variables)
        assert isinstance(variables, list)
        assert isinstance(variables[0], Variable)

        # Store objective and constraint functions
        self.fmodel = model

        # Save options (make sure population is even for pairing purposes)
        self.options = {}
        self.options['restart']     = restart
        self.options['penalty']     = penalty
        self.options['generations'] = maxgen
        self.options['disp']        = True
        self.options['tol']         = tol
        self.options['probability_of_crossover']  = probCross
        self.npop    = population
        self.pmutate = 1.0 / float(self.nvar) # NSGA2 approach
        if not (population%2 == 0): self.npop += 1

        # Initialize design variable matrix
        if isinstance(xinit, np.ndarray):
            self.xinit = xinit.tolist()
        elif isinstance(xinit, list):
            self.xinit = xinit[:]
        else:
            raise ValueError('Initial design variable vector must be a list or numpy array')
        self.x = [[None]*self.nvar for n in xrange(self.npop)]

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
        

    def _generate_population(self, npop):
        # Initialize container
        x = np.inf * np.ones((npop, self.nvar))
            
        # Populate design variables with Latin Hypercube initialization
        for k in xrange(self.nvar):
            x[:,k] = self.variables[k].sample_lhc(npop)
                
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
        if self.options['restart']:
            self._load_restart()
        else:
            # Be sure initial state is included in population
            self.x = self._generate_population(self.npop-1)
            self.x.append( self.xinit )
                
    
    def _evaluate(self):
        # Initialize outputs
        self.obj = np.inf * np.ones((self.npop,))
        self.con = np.inf * np.ones((self.npop,))
        self.objChild = np.inf * np.ones((self.npop,))
        self.conChild = np.inf * np.ones((self.npop,))

        for n in xrange(self.npop):
            self.obj[n], self.con[n] = self.fmodel( self.x[n] )

        # Determine penalty
        coeff = 10.0 ** np.ceil( np.log10(np.abs(self.obj.min())) )
        self.total = self.obj + coeff*self.con
        
        if not (self.xchild is None):
            for n in xrange(self.npop):
                self.objChild[n], self.conChild[n] = self.fmodel( self.xchild[n] )
                
        self.totalChild = self.objChild + coeff*self.conChild


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

        for n in xrange(self.npop):
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
            if np.random.rand() < self.options['probability_of_crossover']:
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

        
    def _rank(self):
        # Rank all designs first
        isort = np.argsort(self.total) if self.options['penalty'] else np.argsort(self.obj)
        self.total = self.total[isort]
        self.obj   = self.obj[isort]
        self.con   = self.con[isort]
        xobj       = self.x[:]
        for n in xrange(len(isort)):
            xobj[n] = self.x[isort[n]][:]
        self.x = xobj[:]

        
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
        for n in xrange(self.npop):
            self.x[n] = xobj[nextgen[n]][:]

        
    def optimize(self):
        # Setup
        self._initialize()
        self._evaluate()

        # Logging initialization
        fhist = open(HISTFILE, 'w')
        fhist.write('Iter\tObjective\t\tConstraint\n')
        
        # Iteration over generations
        ngen       = self.options['generations']
        objHistory = np.empty((ngen,))
        conHistory = np.empty((ngen,))
        iteration  = 1
        nconverge  = 15
        nround     = int( np.round(np.abs(np.log10(self.options['tol']))) )

        def convtest(x):
            return (np.round(np.sum(np.diff(x[-nconverge:])), nround) == 0.0)

        while iteration <= ngen:
            t = time.time()
        
            # Perform evolution for this generation
            self._mating_selection()
            self._crossover()
            self._mutation()
            self._evaluate()
            self._survival_selection()

            # Logging
            fhist.write(str(iteration)+':\t'+
                        '{0:.6f}'.format(self.obj[0])+'\t\t'+
                        '{0:.6f}'.format(self.con[0])+'\n')
            fhist.flush()
            if (iteration%10 == 0): self._write_restart()
            if self.options['disp']:
                print('Generation,',iteration,'complete.  Elapsed:', np.round(time.time() - t,2),'seconds')

            # Store in history
            objHistory[iteration-1] = self.obj[0]
            conHistory[iteration-1] = self.con[0]

            # Check for convergence
            if ( (iteration > nconverge) and
                 convtest(objHistory) and convtest(conHistory) ):
                break

            # Increment counter
            iteration += 1

        # Final logging
        fhist.close()
        self._write_restart()
        #self._write_output()
        
        return (self.x[0], self.obj[0], self.con[0])
