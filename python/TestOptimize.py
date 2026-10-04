import unittest
import TravelingSanta as ts

class MyTestCase(unittest.TestCase):

    def test_returnOnePointPerRunWhenThereNotEnoughCapacity(self):
        points = []
        points.append(ts.Point(2, 40.733034, -86.762233, 10))
        points.append(ts.Point(3, 43.029635, -91.094895, 10))
        points.append(ts.Point(4, 31.180215, 16.596037, 10))

        giftRun = ts.optimize(points, 10)

        self.assertEqual(3, len(giftRun.all))
        self.assertEqual([[2],[3],[4]], sorted(giftRun.all))

if __name__ == '__main__':
    unittest.main()
