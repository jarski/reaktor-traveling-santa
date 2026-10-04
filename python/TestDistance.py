import unittest
import TravelingSanta as TS
import math

class MyTestCase(unittest.TestCase):


    def test_all_distances(self):
        helsinki = TS.Point(1, 60.1100895, 24.8782586, 10)
        newyork = TS.Point(2, 40.697403, -74.1201045, 10)
        all = TS.calculateAllDistances([helsinki, newyork])
        self.assertEqual((3,3), all.shape)
        self.assertTrue(math.isclose(0, all[0, 0], rel_tol=0.0, abs_tol=0.001))
        self.assertEqual(0, all[1, 1])
        self.assertEqual(0, all[2, 2])
        self.assertTrue(math.isclose(892, all[0, 1], rel_tol=0.0, abs_tol=20))
        self.assertTrue(math.isclose(892, all[1, 0], rel_tol=0.0, abs_tol=20))
        self.assertTrue(math.isclose(6372, all[0, 2], rel_tol=0.0, abs_tol=20))
        self.assertTrue(math.isclose(6372, all[2, 0], rel_tol=0.0, abs_tol=20))
        self.assertTrue(math.isclose(6625, all[1, 2], rel_tol=0.0, abs_tol=20))
        self.assertTrue(math.isclose(6625, all[2, 1], rel_tol=0.0, abs_tol=20))

if __name__ == '__main__':
    unittest.main()
