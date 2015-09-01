import unittest
import inspect

from pandas.util.testing import assert_series_equal

import circlesquare
import pandas as pd
import matplotlib.pylab as plt


print 'Testing module at location:', unittest.__file__



class Test_Model_Methods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = circlesquare.CircleSquare()
        cls.model.make_pts(100)

    def test_count_pts(self):
        self.assertEqual(self.model.count_pts(), 100)

    def test_draw(self): #there should be a better way to test plots
        self.model.draw(axes=plt.subplot(1,1,1))

    def test_draw_specific(self):
        self.model.draw(axes=plt.subplot(1,1,1), pts=self.model.pts.index[::3])


class Test_Interface_Creation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = circlesquare.CircleSquare()
        cls.model.make_pts(100)

        cls.interface = cls.model.new_interface('Seeker')
        cls.interface.make_circles(max_area=.01)

    def test_area(self):
        self.assertLess(max(self.interface.circles.area), .01)

    def test_hardening(self):
        """ run 10 rounds of hardening. there should be at least one discovery..."""
        patch_count = self.interface.harden(50, return_patchcount=True)
        self.assertGreater(patch_count, 0)
        self.assertEqual(self.model.rounds_hardened, 50)

    def test_draw(self):
        self.interface.draw(axes=plt.subplot(1,1,1))


    #test that after hardening, the circle colors in a drawing stay the same



class Test_Comparisons(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = circlesquare.CircleSquare()
        cls.model.make_pts(100)

        cls.defense = cls.model.new_interface('Defense')
        cls.defense.make_circles(max_area=.01)

        cls.offense = cls.model.new_interface('Offense')
        cls.offense.make_circles(max_area=.01)

    def test_plot_correlation(self):
        self.offense.plot_correlation(self.defense)

    def test_learn(self):
        self.offense.learn(self.defense)

    def test_get_correlation(self):
        self.offense.get_correlation(self.defense)

        

if __name__ == '__main__':
    unittest.main()
