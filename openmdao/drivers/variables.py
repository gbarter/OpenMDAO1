from __future__ import print_function
import numpy as np

import numbers
real_types = [numbers.Real]
int_types = [numbers.Integral]
real_types.extend([np.float32, np.float64])
int_types.extend([np.int32, np.int64])
real_types = tuple(real_types)
int_types = tuple(int_types)
bool_types = (bool, np.bool_)

def VariableChooser(val, p_low, p_high, continuous=True):
    if isinstance(val, np.ndarray) or isinstance(val, list) or isinstance(val, tuple):
        val = val[0]

    # Order here matters because otherwise integers are part of "numbers.real"
    if isinstance(val, bool_types):
        return BooleanVariable()
    elif isinstance(val, int_types) or (not continuous):
        return IntegerVariable(p_low, p_high)
    elif isinstance(val, real_types):
        return FloatVariable(p_low, p_high)
    else:
        raise ValueError("Unknown variable type: ", val, type(val))
 

def inverseUniformCDF(p, a, b):
    assert isinstance(p, type(np.array([0.]))) or isinstance(p, float)
    return (a + (b-a)*p)


class Variable(object):
    def __init__(self, low=None, high=None):
        self.lower_bound = low
        self.upper_bound = high
        self.eta_c = 10.0 # crossover index
        self.eta_m = 20.0 # mutation index
        
    def sample_rand(self, npts):
        pass
    def sample_lhc(self, npts):
        pass
    def cross(self, x1, x2):
        pass
    def mutate(self, x):
        pass
    def bound(self, x):
        return np.maximum(np.minimum(x, self.upper_bound), self.lower_bound)
    
class BooleanVariable(Variable):
    def __init__(self):
        super(BooleanVariable, self).__init__()

    def sample_rand(self, npts):
        return (np.random.random((npts,)) > 0.5)
    
    def sample_lhc(self, npts):
        y = (np.linspace(0, 1, npts) > 0.5)
        np.random.shuffle(y)
        return y

    def cross(self, x1, x2):
        assert isinstance(x1, bool)
        assert isinstance(x2, bool)
        y1 = x1 and x2
        y2 = x1 or  x2
        return y1, y2
    
    def mutate(self, x):
        assert isinstance(x, bool)
        return (not x)


class FloatVariable(Variable):
    def __init__(self, low=None, high=None):
        super(FloatVariable, self).__init__(low=low, high=high)
        
    def sample_rand(self, npts):
        return np.random.uniform(self.lower_bound, self.upper_bound, size=(npts,))

    def sample_lhc(self, npts):
        segProbBins = np.linspace(0.0, 1.0, npts+1)
        segProbSize = np.diff(segProbBins)[0]
        probPoints  = segProbBins[:-1] + np.random.uniform(0, segProbSize, size=(npts,))
        np.random.shuffle( probPoints )
        return inverseUniformCDF(probPoints, self.lower_bound, self.upper_bound)

    def cross(self, x1, x2):
        y1 = self.bound( min(x1, x2) )
        y2 = self.bound( max(x1, x2) )
        if y1 == y2: return y1, y2
        # Simple
        #c = np.random.uniform(y1, y2, size=(2,))
        #c1, c2 = c[0], c[1]
        # From NSGA2: Simulated binary crossover
        yl = self.lower_bound
        yu = self.upper_bound

        def betaq(dy):
            beta  = 1.0 + (2.0*dy / (y2-y1))
            alpha = 2.0 - beta**(-(self.eta_c+1.0))
            rand_var = np.random.rand()
            if rand_var <= (1.0/alpha):
                betaq = (rand_var*alpha)**(1.0/(self.eta_c+1.0))
            else:
                betaq = (1.0/(2.0 - rand_var*alpha))**(1.0/(self.eta_c+1.0))
            return betaq

        betaq1 = betaq(y1-yl)
        betaq2 = betaq(yu-y2)
        c1 = 0.5*((y1+y2) - betaq1*(y2-y1))
        c2 = 0.5*((y1+y2) + betaq2*(y2-y1))

        c1 = self.bound(c1)
        c2 = self.bound(c2)
        if np.random.rand() <= 0.5:
            return c1, c2
        else:
            return c2, c1

        
    def mutate(self, x):
        # From NSGA2
        y  = x
        yl = self.lower_bound
        yu = self.upper_bound

        delta1 = (y-yl) / (yu-yl)
        delta2 = (yu-y) / (yu-yl)
        rand_var = np.random.rand()
        mut_pow = 1.0 / (self.eta_m + 1.0)
        if rand_var <= 0.5:
            xy     = 1.0 - delta1;
            val    = 2.0*rand_var + (1.0 - 2.0*rand_var) * xy**(self.eta_m+1.0)
            deltaq = val**mut_pow - 1.0
        else:
            xy     = 1.0 - delta2
            val    = 2.0*(1.0 - rand_var) + 2.0*(rand_var-0.5) * xy**(self.eta_m+1.0)
            deltaq = 1.0 - val**mut_pow

        y = y + deltaq*(yu - yl)
        return self.bound(y)
    
    
class IntegerVariable(Variable):
    def __init__(self, low=None, high=None):
        super(IntegerVariable, self).__init__(low=low, high=high)

    def sample_rand(self, npts):
        return np.random.randint(self.lower_bound, self.upper_bound+1, size=(npts,))

    def sample_lhc(self, npts):
        allvals = np.arange(self.lower_bound, self.upper_bound+1)
        probPoints = len(allvals) * np.linspace(0.0, 1.0-1e-10, npts)
        probPoints = probPoints.astype(np.int64)
        np.random.shuffle( probPoints )
        return allvals[ probPoints ]

    def cross(self, x1, x2):
        y1 = int( self.bound( min(x1, x2) ))
        y2 = int( self.bound( max(x1, x2) ))
        if y1 == y2: return y1, y2
        c = np.random.randint(y1, y2+1, size=(2,))
        c1, c2 = c[0], c[1]
        if np.random.rand() <= 0.5:
            return c1, c2
        else:
            return c2, c1

    def mutate(self, x):
        y  = x
        yl = self.lower_bound
        yu = self.upper_bound
        dl = y - yl
        du = yu - y
        try:
            dy = np.random.randint(0, max(dl, du))
        except:
            dy = 0
        if np.random.rand() < 0.5:
            dy = -min(dy, dl)
        else:
            dy = min(dy, du)
        y += dy
        return int(np.round(y))
