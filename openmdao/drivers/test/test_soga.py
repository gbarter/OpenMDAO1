import numpy as np
import numpy.testing as npt
import unittest
import openmdao.drivers.soga as soga
import openmdao.drivers.variables as v


def testCon(x):
    y = np.array(x).sum()
    return ( max(0, (y - 1)) + max(0, (-1 - y)) )

def testObj(x):
    return (np.array(x).sum(), testCon(x))

class TestSOGA(unittest.TestCase):
    def setUp(self):
        self.variables = []
        self.variables.append( v.FloatVariable(-1.0, 1.0) )
        self.variables.append( v.IntegerVariable(0, 10) )
        self.mysoga = soga.SOGA(self.variables, [1/np.pi, 5], testObj, population=3, maxgen=2, probCross=1.0, restart=False, penalty=False, tol=1e-6)
        self.mysoga.pmutate = 1.0

    def testInit(self):
        self.mysoga._initialize()
        
        self.assertEqual(len(self.mysoga.x), self.mysoga.npop)
        for n in xrange(self.mysoga.npop):
            self.assertEqual(len(self.mysoga.x[n]), self.mysoga.nvar)
            for m in xrange(self.mysoga.npop):
                if n == m: continue
                self.assertNotEqual(self.mysoga.x[n], self.mysoga.x[m])

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
        
    def testRank(self):
        self.mysoga._initialize()
        self.mysoga._evaluate()
        xorig    = self.mysoga.x[:]
        objorig  = self.mysoga.obj.copy()
        conorig  = self.mysoga.con.copy()
        totorig  = self.mysoga.total.copy()
        iobj     = np.argsort(objorig)
        itot     = np.argsort(totorig) 
        self.mysoga._rank()

        # Penalty false
        npt.assert_equal(self.mysoga.obj, objorig[iobj])
        npt.assert_equal(self.mysoga.con, conorig[iobj])
        npt.assert_equal(self.mysoga.total, totorig[iobj])
        for n in xrange(self.mysoga.npop):
            self.assertEqual(self.mysoga.x[n], xorig[iobj[n]])

        self.mysoga.options['penalty'] = True
        self.mysoga._rank()
        npt.assert_equal(self.mysoga.obj, objorig[itot])
        npt.assert_equal(self.mysoga.con, conorig[itot])
        npt.assert_equal(self.mysoga.total, totorig[itot])
        for n in xrange(self.mysoga.npop):
            self.assertEqual(self.mysoga.x[n], xorig[itot[n]])
       
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

        
    def testRestart(self):
        self.mysoga._initialize()
        self.mysoga._evaluate()
        self.mysoga._mating_selection()
        self.mysoga._crossover()
        self.mysoga._evaluate()
        self.mysoga._survival_selection()
        self.mysoga._write_restart()

        soga2 = soga.SOGA(self.variables, [0.5, 5], testObj, population=3, restart=True)
        soga2._initialize()
        self.assertEqual(self.mysoga.x, soga2.x)
        
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestSOGA))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
