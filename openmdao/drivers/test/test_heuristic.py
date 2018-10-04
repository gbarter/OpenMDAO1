import numpy as np
import numpy.testing as npt
import unittest
from openmdao.drivers.subplex import Subplex
from openmdao.drivers.soga import SOGA
from openmdao.drivers.sopso import SOPSO
from openmdao.drivers.heuristic import Heuristic
import openmdao.drivers.variables as v


def testCon(x):
    y = np.array(x).sum()
    return ( max(0, (y - 1)) + max(0, (-1 - y)) )

def testObj(x):
    return (np.array(x).sum(), testCon(x))

class TestHeuristic(unittest.TestCase):
    def setUp(self):
        self.variables = []
        self.variables.append( v.FloatVariable(-1.0, 1.0) )
        self.variables.append( v.IntegerVariable(0, 10) )
        self.options = {}
        self.options['population'] = 3
        self.options['generations'] = 2
        self.options['tolerance'] = 1e-6
        self.options['restart'] = False
        self.options['penalty'] = False
        self.options['global_search'] = True
        self.options['penalty_multiplier'] = 1e2
        self.myheur = Heuristic(self.variables, [1/np.pi, 5], testObj, self.options)

    def testInit(self):
        self.myheur._initialize()
        
        self.assertEqual(len(self.myheur.x), self.myheur.npop)
        for n in xrange(self.myheur.npop):
            self.assertEqual(len(self.myheur.x[n]), self.myheur.nvar)
            for m in xrange(self.myheur.npop):
                if n == m: continue
                self.assertNotEqual(self.myheur.x[n], self.myheur.x[m])

    def testEval(self):
        self.myheur._initialize()
        self.myheur._evaluate()
        
        self.assertEqual(self.myheur.con.size, self.myheur.npop)
        self.assertEqual(self.myheur.obj.size, self.myheur.npop)
        self.assertEqual(self.myheur.total.size, self.myheur.npop)
        npt.assert_array_less(0.0, self.myheur.con + 1e-10)
        
    def testRank(self):
        self.myheur._initialize()
        self.myheur._evaluate()
        xorig    = self.myheur.x[:]
        objorig  = self.myheur.obj.copy()
        conorig  = self.myheur.con.copy()
        totorig  = self.myheur.total.copy()
        iobj     = np.argsort(objorig)
        itot     = np.argsort(totorig) 
        self.myheur._rank()

        # Penalty false
        npt.assert_equal(self.myheur.obj, objorig[iobj])
        npt.assert_equal(self.myheur.con, conorig[iobj])
        npt.assert_equal(self.myheur.total, totorig[iobj])
        for n in xrange(self.myheur.npop):
            self.assertEqual(self.myheur.x[n], xorig[iobj[n]])

        self.myheur.options['penalty'] = True
        self.myheur._rank()
        npt.assert_equal(self.myheur.obj, objorig[itot])
        npt.assert_equal(self.myheur.con, conorig[itot])
        npt.assert_equal(self.myheur.total, totorig[itot])
        for n in xrange(self.myheur.npop):
            self.assertEqual(self.myheur.x[n], xorig[itot[n]])

        
    def testRestart(self):
        self.myheur._initialize()
        self.myheur._evaluate()
        self.myheur._write_restart()

        tempOpt = self.options
        tempOpt['restart'] = True
        heur2 = Heuristic(self.variables, [0.5, 5], testObj, tempOpt)
        heur2._initialize()
        self.assertEqual(self.myheur.x, heur2.x)



        

class TestSOGA(unittest.TestCase):
    def setUp(self):
        self.variables = []
        self.variables.append( v.FloatVariable(-1.0, 1.0) )
        self.variables.append( v.IntegerVariable(0, 10) )
        self.options = {}
        self.options['population'] = 4
        self.options['generations'] = 2
        self.options['tolerance'] = 1e-6
        self.options['restart'] = False
        self.options['penalty'] = False
        self.options['global_search'] = True
        self.options['probability_of_crossover'] = 1.0
        self.options['probability_of_mutation'] = 1.0
        self.options['crossover_index'] = 10.0
        self.options['mutation_index'] = 20.0
        self.options['penalty_multiplier'] = 1e2
        self.mysoga = SOGA(self.variables, [1/np.pi, 5], testObj, self.options)

    def testEval(self):
        self.mysoga._initialize()
        self.mysoga._evaluate()
        
        self.assertEqual(self.mysoga.con.size, self.mysoga.npop)
        self.assertEqual(self.mysoga.obj.size, self.mysoga.npop)
        self.assertEqual(self.mysoga.total.size, self.mysoga.npop)
        npt.assert_array_less(0.0, self.mysoga.con + 1e-10)
        npt.assert_equal(np.inf, self.mysoga.objChild)
        npt.assert_equal(np.inf, self.mysoga.conChild)
        npt.assert_equal(np.inf, self.mysoga.totalChild)

        self.mysoga.xchild = self.mysoga.x[:]
        self.mysoga._evaluate()
        npt.assert_array_equal(self.mysoga.conChild, self.mysoga.con)
        npt.assert_array_equal(self.mysoga.objChild, self.mysoga.obj)
        npt.assert_array_equal(self.mysoga.totalChild, self.mysoga.total)

    def testMating(self):
        self.mysoga._initialize()
        self.mysoga._evaluate()
        self.mysoga._mating_selection()

        self.assertEqual(len(self.mysoga.xmate), self.mysoga.npop)
        for n in xrange(self.mysoga.npop):
            self.assertIn(self.mysoga.xmate[n], self.mysoga.x)
            # Make sure there is some diversity among parents
            flag = False
            for m in xrange(self.mysoga.npop):
                try:
                    self.assertNotEqual(self.mysoga.xmate[n], self.mysoga.xmate[m])
                    flag = True
                except: continue
            self.assertTrue(flag)

    def testCrossover(self):
        self.mysoga._initialize()
        self.mysoga._evaluate()
        self.mysoga._mating_selection()
        self.mysoga._crossover()

        self.assertEqual(len(self.mysoga.xchild), self.mysoga.npop)
        for n in xrange(self.mysoga.npop):
            self.assertNotIn(self.mysoga.xchild[n], self.mysoga.xmate)

    def testCombine(self):
        self.mysoga._initialize()
        self.mysoga._evaluate()
        self.mysoga._mating_selection()
        self.mysoga._crossover()
        self.mysoga._evaluate()
        xorig    = self.mysoga.x[:]
        xchild   = self.mysoga.xchild[:]
        objorig  = self.mysoga.obj.copy()
        conorig  = self.mysoga.con.copy()
        totorig  = self.mysoga.total.copy()
        objchild = self.mysoga.objChild.copy()
        conchild = self.mysoga.conChild.copy()
        totchild = self.mysoga.totalChild.copy()
        self.mysoga._combine()

        npt.assert_equal(self.mysoga.obj, np.hstack((objorig, objchild)))
        npt.assert_equal(self.mysoga.con, np.hstack((conorig, conchild)))
        npt.assert_equal(self.mysoga.total, np.hstack((totorig, totchild)))
        self.assertEqual(len(self.mysoga.x), 2*self.mysoga.npop)
        for n in xrange(self.mysoga.npop):
            self.assertEqual(self.mysoga.x[n], xorig[n])
            self.assertEqual(self.mysoga.x[n+self.mysoga.npop], xchild[n])
        self.assertIsNone(self.mysoga.xmate)
        self.assertIsNone(self.mysoga.xchild)
        self.assertIsNone(self.mysoga.objChild)
        self.assertIsNone(self.mysoga.conChild)
        self.assertIsNone(self.mysoga.totalChild)
       
    def testSurvivePenaltyFalse(self):
        self.mysoga._initialize()
        self.mysoga._evaluate()
        self.mysoga._mating_selection()
        self.mysoga._crossover()
        self.mysoga._evaluate()
        xorig    = self.mysoga.x[:]
        xchild   = self.mysoga.xchild[:]
        objorig  = self.mysoga.obj.copy()
        conorig  = self.mysoga.con.copy()
        totorig  = self.mysoga.total.copy()
        objchild = self.mysoga.objChild.copy()
        conchild = self.mysoga.conChild.copy()
        totchild = self.mysoga.totalChild.copy()
        self.mysoga._survival_selection()

        self.assertEqual(len(self.mysoga.x), self.mysoga.npop)
        self.assertEqual(self.mysoga.con.size, self.mysoga.npop)
        self.assertEqual(self.mysoga.obj.size, self.mysoga.npop)
        self.assertEqual(self.mysoga.total.size, self.mysoga.npop)
        npt.assert_array_less(0.0, np.diff(self.mysoga.con)+1e-10) # Ascending
        #npt.assert_array_less(0.0, np.diff(self.mysoga.obj)) # Ascending
        npt.assert_array_less(0.0, np.diff(self.mysoga.total)+1e-10) # Ascending

        
    def testSurvivePenaltyTrue(self):
        self.mysoga.options['penalty'] = True
        self.mysoga._initialize()
        self.mysoga._evaluate()
        self.mysoga._mating_selection()
        self.mysoga._crossover()
        self.mysoga._evaluate()

        self.mysoga._rank() # Doing this one out of order for test purposes
        xorig    = self.mysoga.x[:]
        objorig  = self.mysoga.obj.copy()
        conorig  = self.mysoga.con.copy()
        totorig  = self.mysoga.total.copy()
        # Penalize children
        self.mysoga.objChild += 1e10
        self.mysoga.totalChild += 1e10
        
        self.mysoga._survival_selection()

        self.assertEqual(self.mysoga.x, xorig)
        npt.assert_equal(self.mysoga.con, conorig)
        npt.assert_equal(self.mysoga.obj, objorig)
        npt.assert_equal(self.mysoga.total, totorig)
        npt.assert_array_less(0.0, np.diff(self.mysoga.con)+1e-10) # Ascending
        npt.assert_array_less(0.0, np.diff(self.mysoga.obj)) # Ascending
        npt.assert_array_less(0.0, np.diff(self.mysoga.total)) # Ascending
        



        

class TestSOPSO(unittest.TestCase):
    def setUp(self):
        self.variables = []
        self.variables.append( v.FloatVariable(-1.0, 1.0) )
        self.variables.append( v.IntegerVariable(0, 10) )
        self.options = {}
        self.options['population'] = 3
        self.options['generations'] = 2
        self.options['tolerance'] = 1e-6
        self.options['restart'] = False
        self.options['penalty'] = False
        self.options['global_search'] = True
        self.options['upper_inertia'] = 1.0
        self.options['lower_inertia'] = 0.0
        self.options['cognitive_attraction'] = 1.0
        self.options['social_attraction'] = 1.0
        self.options['penalty_multiplier'] = 1e2
        self.mypso = SOPSO(self.variables, [1/np.pi, 5], testObj, self.options)

    def testLocal(self):
        self.mypso._initialize()
        self.mypso._evaluate()
        self.mypso._update_local()

        npt.assert_equal(self.mypso.obj,   self.mypso.objLocal) 
        npt.assert_equal(self.mypso.con,   self.mypso.conLocal) 
        npt.assert_equal(self.mypso.total, self.mypso.totalLocal)
        self.assertEqual(self.mypso.x, self.mypso.xlocal)

    def testGlobal(self):
        self.mypso._initialize()
        self.mypso._evaluate()
        self.mypso._update_local()
        self.mypso._update_global()

        self.assertEqual(self.mypso.totalGlobal, self.mypso.total.min())
        self.assertIn(self.mypso.xglobal, self.mypso.x)
        

class TestSubplex(unittest.TestCase):
    def setUp(self):
        self.variables = []
        for k in range(9):
            self.variables.append( v.FloatVariable(-1.0, 1.0) )
        self.options = {}
        self.options['population'] = 1
        self.options['generations'] = 2
        self.options['tolerance'] = 1e-6
        self.options['restart'] = False
        self.options['penalty'] = False
        self.options['penalty_multiplier'] = False
        self.options['global_search'] = False
        self.options['adaptive_simplex'] = False
        self.options['penalty_multiplier'] = 1e2
        self.mysub = Subplex(self.variables, np.random.random((9,)), testObj, self.options)

    def testSubspaces(self):
        # From Tom Rowan's thesis
        self.mysub.deltax = np.array([0.7, -0.1, 0.0, 0.01, 1.1, -0.8, 0.2, -0.7, 0.0])
        self.mysub._set_subspaces()
        
        self.assertEqual(len(self.mysub.subspaces), 3)
        
        for m in range(len(self.mysub.subspaces)):
            self.mysub.subspaces[m].sort()
            
        self.assertEqual(self.mysub.subspaces[0], [0,4,5,7])
        self.assertEqual(self.mysub.subspaces[1], [1,6])
        self.assertEqual(self.mysub.subspaces[2], [2,3,8])

        
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestHeuristic))
    suite.addTest(unittest.makeSuite(TestSOGA))
    suite.addTest(unittest.makeSuite(TestSOPSO))
    suite.addTest(unittest.makeSuite(TestSubplex))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
