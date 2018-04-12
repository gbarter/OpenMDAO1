import numpy as np
import numpy.testing as npt
import unittest
import openmdao.drivers.soga as soga
import openmdao.drivers.variables as v

def testObj(x):
    return np.array(x).sum()

def testCon(x):
    y = np.array(x).sum()
    return ( max(0, (y - 1)) + max(0, (-1 - y)) )

class TestSOGA(unittest.TestCase):
    def setUp(self):
        variables = []
        variables.append( v.FloatVariable(-1.0, 1.0) )
        variables.append( v.IntegerVariable(0, 10) )
        self.mysoga = soga.SOGA(variables, testObj, testCon, population=3, maxgen=2, probCross=1.0)

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
        npt.assert_array_less(0.0, self.mysoga.con + 1e-10)
        self.assertIsNone(self.mysoga.conChild)
        self.assertIsNone(self.mysoga.objChild)

        self.mysoga.xchild = self.mysoga.x[:]
        self.mysoga._evaluate()
        npt.assert_equal(self.mysoga.conChild, self.mysoga.con)
        npt.assert_equal(self.mysoga.objChild, self.mysoga.obj)

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
        objchild = self.mysoga.objChild.copy()
        conchild = self.mysoga.conChild.copy()
        self.mysoga._combine()

        npt.assert_equal(self.mysoga.obj, np.hstack((objorig, objchild)))
        npt.assert_equal(self.mysoga.con, np.hstack((conorig, conchild)))
        self.assertEqual(len(self.mysoga.x), 2*self.mysoga.npop)
        for n in xrange(self.mysoga.npop):
            self.assertEqual(self.mysoga.x[n], xorig[n])
            self.assertEqual(self.mysoga.x[n+self.mysoga.npop], xchild[n])
        self.assertIsNone(self.mysoga.xmate)
        self.assertIsNone(self.mysoga.xchild)
        self.assertIsNone(self.mysoga.objChild)
        self.assertIsNone(self.mysoga.conChild)
        
    def testRank(self):
        self.mysoga._initialize()
        self.mysoga._evaluate()
        xorig    = self.mysoga.x[:]
        objorig  = self.mysoga.obj.copy()
        conorig  = self.mysoga.con.copy()
        iobj     = np.argsort(objorig)
        self.mysoga._rank()
        
        npt.assert_equal(self.mysoga.obj, np.sort(objorig))
        npt.assert_equal(self.mysoga.con, conorig[iobj])
        for n in xrange(self.mysoga.npop):
            self.assertEqual(self.mysoga.x[n], xorig[iobj[n]])

    def testSurvive(self):
        self.mysoga._initialize()
        self.mysoga._evaluate()
        self.mysoga._mating_selection()
        self.mysoga._crossover()
        self.mysoga._evaluate()
        xorig    = self.mysoga.x[:]
        xchild   = self.mysoga.xchild[:]
        objorig  = self.mysoga.obj.copy()
        conorig  = self.mysoga.con.copy()
        objchild = self.mysoga.objChild.copy()
        conchild = self.mysoga.conChild.copy()
        self.mysoga._survival_selection()

        self.assertEqual(len(self.mysoga.x), self.mysoga.npop)
        self.assertEqual(self.mysoga.con.size, self.mysoga.npop)
        self.assertEqual(self.mysoga.obj.size, self.mysoga.npop)
        npt.assert_array_less(0.0, np.diff(self.mysoga.con)+1e-10) # Ascending
        npt.assert_array_less(0.0, np.diff(self.mysoga.obj)) # Ascending
        
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestSOGA))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
