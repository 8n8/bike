import numpy as np
import obstacleMap as o
import unittest

class testMain(unittest.TestCase):
   
   def testTime(self):
       a = np.zeros((1,1))
       self.assertEqual(o.main(a,a,a,a,1).shape, (100,100)) 

unittest.main()
