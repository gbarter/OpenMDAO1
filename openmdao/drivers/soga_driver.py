"""
OpenMDAO Sinlge Object Genetive Algorithm driver for mixed-integer problems
"""

from __future__ import print_function

from six import itervalues, iteritems
from six.moves import range

import numpy as np
from soga import SOGA
from variables import VariableChooser

from openmdao.core.driver import Driver
from openmdao.util.record_util import create_local_meta, update_local_meta

_optimizers = ['SOGA']

class SOGADriver(Driver):
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
        super(SOGADriver, self).__init__()

        # What we support
        self.supports['inequality_constraints'] = True
        self.supports['equality_constraints'] = True
        self.supports['multiple_objectives'] = False
        self.supports['active_set'] = False
        self.supports['gradients'] = False

        # User Options
        self.options.add_option('optimizer', 'SOGA', values=_optimizers,
                                desc='Name of optimizer to use')
        self.options.add_option('restart', False,
                                desc='Whether to restart from previous soga.restart file')
        self.options.add_option('penalty', True,
                                desc='Whether to consider constraints as penalty on objective function')
        self.options.add_option('population', 200, lower=2.0,
                                desc='Number of designs to carry through each generation')
        self.options.add_option('generations', 200, lower=0.0,
                                desc='Number of generations to evolve each design')
        self.options.add_option('probability_of_crossover', 0.9, lower=0.0, upper=1.0,
                                desc='Probability of mating between two individuals.')
        self.options.add_option('disp', True,
                                desc='Set to False to prevent printing of Scipy '
                                'convergence messages')
        self.options.add_option('tol', 1e-6, lower=1e-15, upper=1e-1,
                                desc='Tolerance for termination.')
        

        self.variables = None
        self.nvar      = None
        self.metadata  = None
        self.exit_flag = 0

        
    def prerun(self, problem):

        # Metadata Setup
        #opt = self.options['optimizer']
        self.metadata = create_local_meta(None, 'SOGA') #opt
        self.iter_count = 0
        update_local_meta(self.metadata, (self.iter_count,))

        # Initial Run
        with problem.root._dircontext:
            problem.root.solve_nonlinear(metadata=self.metadata)

        pmeta = self.get_desvar_metadata()
        self.params = list(pmeta)
        con_meta = self.get_constraint_metadata()

        # Size Problem
        nparam = 0
        for param in itervalues(pmeta):
            nparam += param['size']
        x_init = np.empty(nparam)

        # Initial Parameters
        i = 0
        self.variables = []
        for name, val in iteritems(self.get_desvars()):
            size = pmeta[name]['size']
            x_init[i:i+size] = val
            i += size

            # Bounds and variable vector
            meta_low  = pmeta[name]['lower']
            meta_high = pmeta[name]['upper']

            meta_cont = pmeta[name]['continuous']

            for j in range(size):
                
                if isinstance(meta_low, np.ndarray):
                    p_low = meta_low[j]
                else:
                    p_low = meta_low

                if isinstance(meta_high, np.ndarray):
                    p_high = meta_high[j]
                else:
                    p_high = meta_high

                self.variables.append( VariableChooser(val, p_low, p_high, continuous=meta_cont) )
        self.nvar = len(self.variables)

        # Constraints
        self.constraints = []
        i = 0
        for name, meta in con_meta.items():
            self.constraints.append(name)
            dblcon = meta['upper'] is not None and meta['lower'] is not None
            if dblcon:
                name = '2bl-' + name
                self.constraints.append(name)

        return x_init

    
    def postrun(self, xresult, fmin, fcon):
        # Re-run optimal point so that framework is left in the right final state
        self._unpack(xresult)
        with self.root._dircontext:
            self.root.solve_nonlinear(metadata=self.metadata)
        
        self.exit_flag = 1

        if self.options['disp']:
            print('Optimization Complete')
            print('-'*35)
            print('Objective Function: ', str(fmin))
            print('-'*35)
            print('Cumulative Constraints: ', str(fcon))
            print('-'*35)

            
    def run(self, problem):
        """Optimize the problem

        Args
        ----
        problem : `Problem`
            Our parent `Problem`.
        """
        # Prep and get initial starting design
        x_init = self.prerun(problem)
        
        # Optimize
        optimizer = SOGA(self.variables, x_init, self._model,
                         tol=self.options['tol'],
                         restart=self.options['restart'],
                         penalty=self.options['penalty'],
                         population=self.options['population'],
                         maxgen=self.options['generations'],
                         probCross=self.options['probability_of_crossover'])
        xresult, fmin, fcon = optimizer.optimize()

        # Process results
        self.postrun(xresult, fmin, fcon)

        
    def _unpack(self, x_new):
        # Pass in new parameters
        i = 0
        for name, meta in self.get_desvar_metadata().items():
            size = meta['size']
            self.set_desvar(name, np.array( x_new[i:i+size] ))
            i += size
            
        # Update meta data
        self.iter_count += 1
        update_local_meta(self.metadata, (self.iter_count,))

        
    def _model(self, xin):
        """ Function that evaluates and returns the objective function. Model
        is executed here.

        Args
        ----
        xin : ndarray
            Array containing parameter values at new design point.

        Returns
        -------
        float
            Value of the objective function evaluated at the new design point.
        """
        # Set design variable vectors
        if isinstance(xin, list) and (len(xin) == self.nvar):
            self._unpack(xin)

        # Run model
        with self.root._dircontext:
            self.root.solve_nonlinear(metadata=self.metadata)

        # Get the objective function evaluations
        for name, obj in self.get_objectives().items():
            f_new = obj[0]
            break

        # Get all constraints and sum up the violations
        conDict = self.get_constraints()
        consum  = 0.0

        for name in self.constraints:
            # Catch the double-sided addition we created at initialization
            if name.startswith('2bl-'):
                myname = name[4:]
                dbl_side = True
            else:
                myname = name
                dbl_side = False

            # Get meta data including target values
            meta = self._cons[myname]

            # Get cached constraint value from when model was executed
            val = conDict[myname]
            
            # Get constraint violations (defined as positive here)
            bound = meta['equals']
            if bound is not None: # Equality constraints
                temp = np.abs(bound - val)
                norm = bound

            else:
                # Inequality constraints
                upper = meta['upper']
                lower = meta['lower']
                if lower is None or dbl_side:
                    temp = val - upper
                    norm = upper
                else:
                    temp = lower - val
                    norm = lower

            # Zero out constraint compliance so we don't drown out violations with excessive margin
            temp = np.maximum(temp, 0.0)

            # Make relative to constraint target
            if isinstance(norm, np.ndarray): 
                norm[norm==0] = 1.0
            elif norm == 0.0:
                norm = 1.0
            temp /= np.abs(norm)
            
            # Add to cumulative violation score
            if isinstance(temp, np.ndarray): temp = temp.sum()
            consum += temp
        
        # Record after getting obj and constraints to assure it has been
        # gathered in MPI.
        self.recorders.record_iteration(self.root, self.metadata)

        return f_new, consum
