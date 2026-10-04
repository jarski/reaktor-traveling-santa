import unittest
import TravelingSanta as ts
import math


class MyTestCase(unittest.TestCase):

    def test_amount_of_points(self):
        points = ts.readPoints()
        self.assertEqual(10000, len(points))

    def test_first_point(self):
        first = ts.readPoints()[0]
        latitude = math.radians(40.733034)
        longitude = math.radians(-86.762233)
        x = ts.R * math.cos(latitude) * math.cos(longitude)
        y = ts.R * math.cos(latitude) * math.sin(longitude)
        z = ts.R * math.sin(latitude)
        self.assertEqual(2, first.id)
        self.assertEqual(342917, first.weight)
        self.assertEqual(x, first.vector[0])
        self.assertEqual(y, first.vector[1])
        self.assertEqual(z, first.vector[2])

    def test_last_point(self):
        points = ts.readPoints()
        last = points[len(points)-1]
        latitude = math.radians(-22.55115)
        longitude = math.radians(-63.770996)
        x = ts.R * math.cos(latitude) * math.cos(longitude)
        y = ts.R * math.cos(latitude) * math.sin(longitude)
        z = ts.R * math.sin(latitude)
        self.assertEqual(10001, last.id)
        self.assertEqual(663913, last.weight)
        self.assertEqual(x, last.vector[0])
        self.assertEqual(y, last.vector[1])
        self.assertEqual(z, last.vector[2])


if __name__ == '__main__':
    unittest.main()
