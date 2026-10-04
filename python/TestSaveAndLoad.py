import unittest
import TravelingSanta as ts

class MyTestCase(unittest.TestCase):

    def test_returnsAllPoints(self):
        points = ts.readPoints()[:100]
        giftRun = ts.Ant(points, 10000000).findSolution()
        ts.saveBest(giftRun)
        loaded = ts.loadBest()
        self.assertEqual(len(giftRun.all), len(loaded.all))
        self.assertEqual(giftRun.totalDistance, loaded.totalDistance)
        self.assertEqual(sorted(giftRun.all), sorted(loaded.all))

if __name__ == '__main__':
    unittest.main()
