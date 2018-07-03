import numpy as np
import numpy.testing as npt
import unittest
import openmdao.drivers.variables as v

NPTS = 100
eta_c = 10.0
eta_m = 20.0

class TestFloat(unittest.TestCase):
    def setUp(self):
        self.myvar = v.FloatVariable(0.0, 10.0)

    def testRand(self):
        x = self.myvar.sample_rand(NPTS)
        self.assertEqual(x.shape, (NPTS,) )
        npt.assert_array_less(x, self.myvar.upper_bound)
        npt.assert_array_less(self.myvar.lower_bound, x)

    def testLHC(self):
        x = self.myvar.sample_lhc(NPTS)
        self.assertEqual(x.shape, (NPTS,) )
        npt.assert_array_less(x, self.myvar.upper_bound)
        npt.assert_array_less(self.myvar.lower_bound, x)
        hc,_ = np.histogram(x, bins=np.linspace(self.myvar.lower_bound, self.myvar.upper_bound, NPTS+1))
        npt.assert_equal(hc, np.ones(NPTS))

    def testCross(self):
        c1, c2 = self.myvar.cross(1.0, 9.0, eta_c)
        self.assertLessEqual(c1, self.myvar.upper_bound)
        self.assertLessEqual(c2, self.myvar.upper_bound)
        self.assertLessEqual(self.myvar.lower_bound, c1)
        self.assertLessEqual(self.myvar.lower_bound, c2)

        c1, c2 = self.myvar.cross(1.0, 11.0, eta_c)
        self.assertLessEqual(c1, self.myvar.upper_bound)
        self.assertLessEqual(c2, self.myvar.upper_bound)
        self.assertLessEqual(self.myvar.lower_bound, c1)
        self.assertLessEqual(self.myvar.lower_bound, c2)

        c1, c2 = self.myvar.cross(5.01, 5.02, eta_c)
        self.assertLessEqual(c1, self.myvar.upper_bound)
        self.assertLessEqual(c2, self.myvar.upper_bound)
        self.assertLessEqual(self.myvar.lower_bound, c1)
        self.assertLessEqual(self.myvar.lower_bound, c2)
        self.assertLessEqual(np.abs(c1-5.015), 0.01)
        self.assertLessEqual(np.abs(c2-5.015), 0.01)
        
        c1, c2 = self.myvar.cross(5.0, 5.0, eta_c)
        self.assertEqual(c1, 5.0)
        self.assertEqual(c2, 5.0)

    def testMutate(self):
        for i in np.linspace(self.myvar.lower_bound, self.myvar.upper_bound, 11):
            x = self.myvar.mutate(i, eta_m)
            self.assertLessEqual(x, self.myvar.upper_bound)
            self.assertLessEqual(self.myvar.lower_bound, x)

    def testVelocity(self):
        dV = self.myvar.velocity_update(5.0, 5.0, 5.0, 0.5, 1.0)
        self.assertEqual(dV, 0.0)
        dV = self.myvar.velocity_update(5.0, 6.0, 5.0, 0.0, 1.0)
        self.assertEqual(dV, 0.0)
        dV = self.myvar.velocity_update(5.0, 5.0, 6.0, 0.5, 0.0)
        self.assertEqual(dV, 0.0)
        
        dV = self.myvar.velocity_update(5.0, 5.0, 4.0, 1.0, 1.0)
        self.assertLess(dV, 0.0)
        dV = self.myvar.velocity_update(5.0, 4.0, 4.0, 1.0, 1.0)
        self.assertLess(dV, 0.0)
    
        dV = self.myvar.velocity_update(5.0, 5.0, 6.0, 1.0, 1.0)
        self.assertGreater(dV, 0.0)
        dV = self.myvar.velocity_update(5.0, 6.0, 6.0, 1.0, 1.0)
        self.assertGreater(dV, 0.0)
    
    def testPosition(self):
        self.assertEqual(self.myvar.position_update(5.0,  1.0), 6.0)
        self.assertEqual(self.myvar.position_update(5.0, -1.0), 4.0)
        self.assertEqual(self.myvar.position_update(5.0,  10.0), self.myvar.upper_bound)
        self.assertEqual(self.myvar.position_update(5.0, -10.0), self.myvar.lower_bound)

        
class TestInt(unittest.TestCase):
    def setUp(self):
        self.myvar = v.IntegerVariable(0, 4)

    def testRand(self):
        x = self.myvar.sample_rand(NPTS)
        self.assertEqual(x.shape, (NPTS,) )
        npt.assert_array_less(x, self.myvar.upper_bound+0.1)
        npt.assert_array_less(self.myvar.lower_bound-0.1, x)

    def testLHC(self):
        x = self.myvar.sample_lhc(NPTS)
        self.assertEqual(x.shape, (NPTS,) )
        npt.assert_array_less(x, self.myvar.upper_bound+0.1)
        npt.assert_array_less(self.myvar.lower_bound-0.1, x)
        hc,_ = np.histogram(x, bins=np.arange(self.myvar.lower_bound, 1.1+self.myvar.upper_bound)-0.1)
        npt.assert_equal(hc, (NPTS/len(hc))*np.ones(len(hc)))

    def testCross(self):
        c1, c2 = self.myvar.cross(1, 5, eta_c)
        self.assertLessEqual(c1, self.myvar.upper_bound)
        self.assertLessEqual(c2, self.myvar.upper_bound)
        self.assertLessEqual(self.myvar.lower_bound, c1)
        self.assertLessEqual(self.myvar.lower_bound, c2)
        self.assertLessEqual(c1, 5)
        self.assertLessEqual(c2, 5)
        self.assertLessEqual(1, c1)
        self.assertLessEqual(1, c2)

        c1, c2 = self.myvar.cross(1, 11, eta_c)
        self.assertLessEqual(c1, self.myvar.upper_bound)
        self.assertLessEqual(c2, self.myvar.upper_bound)
        self.assertLessEqual(self.myvar.lower_bound, c1)
        self.assertLessEqual(self.myvar.lower_bound, c2)
        self.assertLessEqual(1, c1)
        self.assertLessEqual(1, c2)
        
        c1, c2 = self.myvar.cross(3, 3, eta_c)
        self.assertEqual(c1, 3)
        self.assertEqual(c2, 3)
        
        c1, c2 = self.myvar.cross(5, 5, eta_c)
        self.assertEqual(c1, 4)
        self.assertEqual(c2, 4)

    def testMutate(self):
        for i in np.arange(self.myvar.lower_bound, self.myvar.upper_bound+1):
            x = self.myvar.mutate(i, eta_m)
            self.assertLessEqual(x, self.myvar.upper_bound)
            self.assertLessEqual(self.myvar.lower_bound, x)

    
    def testPosition(self):
        self.assertEqual(self.myvar.position_update(3,  1), 4)
        self.assertEqual(self.myvar.position_update(3, -1), 2)
        self.assertEqual(self.myvar.position_update(3,  10), self.myvar.upper_bound)
        self.assertEqual(self.myvar.position_update(3, -10), self.myvar.lower_bound)
            

class TestBoolean(unittest.TestCase):
    def setUp(self):
        self.myvar = v.BooleanVariable()

    def testRand(self):
        x = self.myvar.sample_rand(NPTS)
        self.assertEqual(x.shape, (NPTS,) )

    def testLHC(self):
        x = self.myvar.sample_lhc(NPTS)
        self.assertEqual(x.shape, (NPTS,) )
        self.assertEqual(np.count_nonzero(x), 0.5*NPTS)
        self.assertEqual(np.count_nonzero(np.logical_not(x)), 0.5*NPTS)

    def testCross(self):
        c1, c2 = self.myvar.cross(False, False, eta_c)
        self.assertFalse(c1)
        self.assertFalse(c2)

    def testMutate(self):
        x = self.myvar.mutate(True, eta_m)
        self.assertFalse(x)
        x = self.myvar.mutate(False, eta_m)
        self.assertTrue(x)

    def testVelocity(self):
        dV = self.myvar.velocity_update(True, True, True, 0.5, 1.0)
        self.assertEqual(dV, 0.0)
        dV = self.myvar.velocity_update(True, False, True, 0.0, 1.0)
        self.assertEqual(dV, 0.0)
        dV = self.myvar.velocity_update(True, True, False, 0.5, 0.0)
        self.assertEqual(dV, 0.0)
        
        dV = self.myvar.velocity_update(False, False, True, 1.0, 1.0)
        self.assertGreater(dV, 0.0)
        dV = self.myvar.velocity_update(False, True, True, 1.0, 1.0)
        self.assertGreater(dV, 0.0)
    
        dV = self.myvar.velocity_update(True, True, False, 1.0, 1.0)
        self.assertLess(dV, 0.0)
        dV = self.myvar.velocity_update(True, False, False, 1.0, 1.0)
        self.assertLess(dV, 0.0)
    
    def testPosition(self):
        self.assertEqual(self.myvar.position_update(True,  0.51), True)
        self.assertEqual(self.myvar.position_update(True, -0.51), False)
        self.assertEqual(self.myvar.position_update(False,  0.51), True)
        self.assertEqual(self.myvar.position_update(False, -0.51), False)
        
            
class TestChooser(unittest.TestCase):
    def testChooser(self):
        x = v.VariableChooser(5.0, 1.0, 10.0)
        self.assertIsInstance(x, v.FloatVariable)
        x = v.VariableChooser([5.0], 1.0, 10.0)
        self.assertIsInstance(x, v.FloatVariable)
        x = v.VariableChooser((5.0,), 1.0, 10.0)
        self.assertIsInstance(x, v.FloatVariable)
        x = v.VariableChooser(np.array([5.0]), 1.0, 10.0)
        self.assertIsInstance(x, v.FloatVariable)

        x = v.VariableChooser(5.0, 1.0, 10.0, continuous=False)
        self.assertIsInstance(x, v.IntegerVariable)
        x = v.VariableChooser([5.0], 1.0, 10.0, continuous=False)
        self.assertIsInstance(x, v.IntegerVariable)
        x = v.VariableChooser((5.0,), 1.0, 10.0, continuous=False)
        self.assertIsInstance(x, v.IntegerVariable)
        x = v.VariableChooser(np.array([5.0]), 1.0, 10.0, continuous=False)
        self.assertIsInstance(x, v.IntegerVariable)

        x = v.VariableChooser(5, 1, 10)
        self.assertIsInstance(x, v.IntegerVariable)
        x = v.VariableChooser([5], 1, 10)
        self.assertIsInstance(x, v.IntegerVariable)
        x = v.VariableChooser((5,), 1, 10)
        self.assertIsInstance(x, v.IntegerVariable)
        x = v.VariableChooser(np.array([5]), 1, 10)
        self.assertIsInstance(x, v.IntegerVariable)

        x = v.VariableChooser(True, None, None)
        self.assertIsInstance(x, v.BooleanVariable)
        x = v.VariableChooser([True], None, None)
        self.assertIsInstance(x, v.BooleanVariable)
        x = v.VariableChooser((True,), None, None)
        self.assertIsInstance(x, v.BooleanVariable)
        x = v.VariableChooser(np.array([True]), None, None)
        self.assertIsInstance(x, v.BooleanVariable)

        
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestFloat))
    suite.addTest(unittest.makeSuite(TestInt))
    suite.addTest(unittest.makeSuite(TestBoolean))
    suite.addTest(unittest.makeSuite(TestChooser))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
