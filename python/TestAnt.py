import unittest
import TravelingSanta as ts
import math
from functools import reduce
from timeit import default_timer as timer
import multiprocessing


class MyTestCase(unittest.TestCase):

    # 1. 7.706s
    # 2. 7.942s - calculate total distance
    # 3. 55.621s - choose closest point as next
    # 4. 3.922 + 19.201 - calulate all distances to numpy array
    # - 3 -> 23s
    # - 4 -> 24s
    # - 5 -> 31s
    # - 6 -> 38s
    # - 10 -> 70s
    def test_returnsAllPoints(self):
        points = ts.readPoints()

        start = timer()
        allDistances = ts.calculateAllDistances(points)
        end = timer()
        print("time: " + str(end - start))

        start = timer()
        jobs = []
        n = 4
        queue = multiprocessing.Queue()
        for i in range(n):
            ant = ts.Ant(points, 10000000, allDistances)
            process = multiprocessing.Process(target=ant.findSolution,
                                              args=(queue,))
            jobs.append(process)
        for job in jobs:
            job.start()
        print('started')

        for i in range(n):
            self.assertEqual(10000, len(reduce(lambda x, y: x + y, queue.get().all)))
        end = timer()
        print("time: " + str(end - start))

    def test_CalculateTotalDistance(self):
        points = []
        points.append(ts.Point(2, 60.1100895, 24.8782586, 10)) # Korvatunturi - Helsinki ~892
        points.append(ts.Point(3, 40.697403, -74.1201045, 10)) # Korvatunturi - NewYork ~6372
        # total distance ~ 2*892 + 2*6372 = 14528 (+/- 80)

        giftRun = ts.optimize(points, 10)

        self.assertTrue(math.isclose(14528, giftRun.totalDistance, rel_tol=0.0, abs_tol=80))

if __name__ == '__main__':
    unittest.main()
