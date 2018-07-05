from __future__ import print_function
import numpy as np
from variables import Variable    
from heuristic import Heuristic

class SOPSO(Heuristic):
    def __init__(self, variables=None, xinit=None, model=None, options=None):
        Heuristic.__init__(self, variables, xinit, model, options) # don't use super because of multiple inheritance confusion later

        # Local bests and velocity
        self.xlocal     = [[None]*self.nvar for n in range(self.npop)]
        self.conLocal   = np.inf * np.ones(self.npop)
        self.objLocal   = np.inf * np.ones(self.npop)
        self.totalLocal = np.inf * np.ones(self.npop)
        self.velocity   = np.zeros( (self.npop, self.nvar) )

            
    def _update_local(self):
        # Find personal locals
        if self.options['penalty']:
            idx = np.where(self.total < self.totalLocal)[0]
        else:
            idx = np.where(self.obj < self.objLocal)[0]

        # Update personal local vectors
        if idx.size > 0:
            self.objLocal[idx]   = self.obj[idx]
            self.conLocal[idx]   = self.con[idx]
            self.totalLocal[idx] = self.total[idx]
            for k in idx:
                self.xlocal[k]   = self.x[k][:]

        
    def _update_global(self):
        # Find best performer index
        ibest  = np.argmin(self.totalLocal) if self.options['penalty'] else np.argmin(self.obj)

        # Should we update global bests?
        if self.objGlobal is None:
            updateFlag = True
        else:
            updateFlag = (self.total[ibest] < self.totalGlobal) if self.options['penalty'] else (self.obj[ibest] < self.objGlobal)

        if updateFlag:
            self.objGlobal   = self.obj[ibest]
            self.conGlobal   = self.con[ibest]
            self.totalGlobal = self.total[ibest]
            self.xglobal     = self.x[ibest][:]


    def _update_velocity_and_position(self, k_iter):
        # Constants that control convergence and knowledge of swarm
        cL = self.options['cognitive_attraction'] # local
        cG = self.options['social_attraction'] # global 
       
        # Update particle inertia based on global iteration progress (start higher)
        ngen = float( self.options['generations'] )
        pinertia = np.interp(float(k_iter)/ngen, [0.0, 1.0], [self.options['upper_inertia'], self.options['lower_inertia']])

        # Scale all velocities for this iteration
        self.velocity *= pinertia

        # Get social updates to velocity and update particle position
        for k in range(self.nvar):
            xG = self.xglobal[k]
            for n in range(self.npop):
                x  = self.x[n][k]
                xL = self.xlocal[n][k]
                if xL is None: xL = x
                
                # Update velocity (can bound if want to)
                self.velocity[n][k] += self.variables[k].velocity_update(x, xL, xG, cL, cG)

                # Update particle position and zero out velocity if we hit bounds
                guess = x + self.velocity[n][k]
                self.x[n][k] = self.variables[k].position_update(x, self.velocity[n][k])
                if np.abs(self.x[n][k] - guess) > 0.5:
                    self.velocity[n][k] = 0.0

        
    def _iterate(self, kiter):
        self._update_local()
        self._update_global()
        self._update_velocity_and_position(kiter)
        self._evaluate()
