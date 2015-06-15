
# coding: utf-8

# This notebook will build up the common elements of the circles and squares model, making them accessible to a variety of different analyses

# In[106]:

import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from matplotlib.pylab import Rectangle

class circlesquare:
    """Represent the vulnerabilities of a piece of software as a circle/square model."""
    
    def __init__(self):
        self.rounds_hardened = 0
    
    def make_circles(self, initial_dist=np.random.rand, num_circles=1000, max_area=.005):
        """
            Generate circles representing the vulnerabilities according to
            the initial distribution passed into this function.
            
            initial_dist -- a function which when called returns a single value from a distribution
            
            todo: In the future, we may decide to make this an 'add circles' function, so that
            we can add additional vulnerabilities to an existing piece of software (ie, feature addition)
        """
        xmin, xmax, ymin, ymax = 0, 1, 0, 1

        xc = (xmax - xmin) * np.random.random(num_circles) + xmin
        yc = (ymax - ymin) * np.random.random(num_circles) + ymin
        pts = gpd.GeoSeries([Point(x, y) for x, y in zip(xc, yc)])

        self.circles = pts.apply(lambda x: x.buffer(np.sqrt(max_area * initial_dist())))
    
    def seek(self, rounds):
        """ Look for some vulnerabilities, return a list telling us which of the remaining ones we've found """    
        rounds = int(rounds) 
        searchpoints = gpd.GeoSeries([Point(np.random.rand(2)) for _ in range(rounds)])
        return self.circles.intersects(searchpoints.unary_union)
    
    def patch(self, patch_array):
        """
            Takes a pandas Series, index corresponding to the index of circles, values true or false to patch
        """
        count_before = self.count_circles()
        self.circles = self.circles[patch_array == False]
        count_after = self.count_circles()
        return count_before-count_after
    
    def harden(self, rounds):
        """ Perform single-actor seek and patch. 
            Returns a tuple:
            -- the number of rounds completed (should be the 'floor' of 'rounds')
            -- the number of vulnerabilities patched
        """
        rounds = int(rounds)
        self.rounds_hardened += rounds
        return rounds, self.patch(self.seek(rounds))

    def count_circles(self):
        """ How many circles remain? """
        return len(self.circles)
    
    def plot(self, axes, circles=None, **kwargs):
        """ 
            Plot a geoseries of circles. if they are not defined, plot all of the ones that the 
            object currently has. format nicely.
            
            In theory, any keyword argument that geoseries plot command accepts, we should be able to include.
        """
        if circles is None:
            circles = self.circles
            
        circles.plot(axes=axes, **kwargs)
        axes.axis('off')
        axes.set_aspect('equal')
        axes.set_yticks([])
        axes.set_xticks([])
        axes.set_xlim(-.1, 1.1)
        axes.set_ylim(-.1, 1.1)
        axes.add_patch(Rectangle((0,0), 1, 1, facecolor="grey", alpha=.25, zorder=0))


# In[107]:

# %pylab inline
# model = circlesquare()


# In[108]:

# model.make_circles(num_circles=15)
# ax = plt.subplot(1,1,1)
# model.plot(ax)


# In[109]:

# model.make_circles()
# model.harden(10)


# In[ ]:



