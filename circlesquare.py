"""
This module contains a basic agent-based Vulnerability Discovery Model, as described in the
paper `Lessons for Bug Bounty Programs from Vulnerability Discovery Simulation`

James Houghton
houghton@mit.edu
August 2015
"""

import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from matplotlib.pylab import Rectangle
import matplotlib.pylab as plt
from scipy.stats.stats import pearsonr
import pandas as pd
import random

class CircleSquare(object):
    """Represent the vulnerabilities of a piece of software.
    The software itself is abstracted as a unit square within which
    vulnerabilities are scattered randomly.

    Assumptions
    -----------
    - Vulnerabilities are present in the software when it is released
    - No new vulns are introduced during the patching process
    -
    """

    def __init__(self):
        self.rounds_hardened = 0
        self.pts = None #to be defined in `make_pts`

    def make_pts(self, n_pts=1000):
        """Generate the points which will represent the vulnerabilities.

        Parameters
        ----------
        n_pts: integer
           The number of points (vulnerabilities) that will be added
           to the model.

        """
        xs = np.random.random(n_pts)
        ys = np.random.random(n_pts)
        self.pts = gpd.GeoSeries([Point(x, y) for x, y in zip(xs, ys)])

    def count_pts(self):
        """ How many vulnerabilities remain? """
        return len(self.pts)

    def new_interface(self, name):
        """ Create a new interface to the software,
        representing a particular team or individual's skill set.

        Parameters
        ----------
        name: string
            The name of the team/individual that the interface represents.
            This is used mostly for labeling plots.

        Returns
        -------
        interface: Interface Object
           The interface object built upon this underlying CircleSquare object.

        """
        return self.Interface(self, name)

    def draw(self, axes=None, pts=None):
        """ Draw the components of the underlying fundamental model,
        including the grey square and points representing vulnerabilities.

        Parameters
        ----------
        axes: matplotlib axis object,
           These are the axes which to which we will add components

        pts: None or list of integers
           `None` plots all points in the product
           A list of indicies plots the points which correspond to those values

        Demonstration
        -------------
        >>> model.draw(axes=plt.subplot(1,1,1))
        >>> model.draw(axes=plt.subplot(1,1,1), [0,1,2,3])

        """
        axes.add_patch(Rectangle((0, 0), 1, 1, facecolor="grey", alpha=.25, zorder=0))

        if isinstance(pts, list):
            plotpts = self.pts.loc[pts]
        else:
            plotpts = self.pts

        xs = plotpts.apply(lambda p: p.coords.xy[0][0])
        ys = plotpts.apply(lambda p: p.coords.xy[1][0])

        axes.plot(xs, ys, 'k.')

        axes.axis('off')
        axes.set_aspect('equal')
        axes.set_yticks([])
        axes.set_xticks([])
        axes.set_xlim(-.1, 1.1)
        axes.set_ylim(-.1, 1.1)


    class Interface():
        """ Represent the view a particular actor has into the
        underlying software and vulnerability set. The likelihood that
        an actor will discover a particular vulnerability is represented
        as the area of a circle surrounding

        Circles in this view are the actor-spacific discovery profiles of each bug.

        Assumptions
        -----------
        - The area of each circle is a proxy for the likelihood that its coresponding
        vulnerability will be found in the next round of searching by the relevant actor.
        - Different actors have different views or interfaces to the underlying product,
        whose circles may vary in size from one another.

        """
        def __init__(self, circlesquare, name):
            # Maintain a local reference to the original class
            self.circlesquare = circlesquare
            self.name = name
            self.circles = None #to be defined in `make_circles`

        def learn(self, other, frac=.1, mean_weight=.01):
            """ Learn from another interface.

            Each actor has the opportunity to learn from another actor. They do this
            by increasing the likelihood of discovering a particular vulnerability,
            by a fraction (possibly greater than one) of the difference in the discovery
            profile of the vulnerabiliy as presented to the two actors.

            Parameters
            ----------
            other: Interface object
               A second interface object with the same underlying CircleSquare object

            frac: float [0,1]
               The fraction of points that the learner *could* learn from that they
            actually *do* learn from

            mean_weight: float [0, infinity)
               The amount of difference that the learner gleans from the teacher, on average

            Returns
            -------
            num_realized_opps: int
               Number of vulnerabilities for which the present actor improved their
               chances, by learning from the other actor.

            TODO
            ----
            Consider using distributions other than the exponential to model learning gains.

            """
            # opportunities for learning exist when the other has a larger circle for a vuln than
            # the self
            opps = other.circles[other.circles.area > self.circles.area].index
            num_opps = len(opps)

            # not all opportunities are realized in a given round of learning
            realized_opps = sorted(random.sample(opps, int(frac*num_opps)))
            num_realized_opps = len(realized_opps)

            # want to weight the potential learning with some distribution
            weight = np.random.exponential(mean_weight, num_realized_opps)

            # the new area of the circle/discovery profile is a weighted average of the other
            # and the self areas
            new_areas = (weight * other.circles.loc[realized_opps].area +
                         (1-weight) * self.circles.loc[realized_opps].area)

            new_radii = np.sqrt(new_areas/np.pi)

            update_pts = pd.DataFrame(self.circlesquare.pts[realized_opps], columns=['point'])
            update_pts['index'] = update_pts.index

            new_circles = update_pts.apply(lambda x: x['point'].buffer(new_radii[x['index']]),
                                           axis=1)

            self.circles[realized_opps] = new_circles
            return num_realized_opps

        def get_correlation(self, other_interface):
            """ Calculate the pearson correlation between the present interface
            and another interface to the same product.

            Parameters
            ----------
            other_interface: Interface object
               The interface object must be built upon the same underlying 'product'
               as the present interface

            Returns
            -------
            c: float
               Pearson's correlation coefficient,

            p: float
               2-tailed p-value

            """
            return pearsonr(self.circles.area, other_interface.circles.area)

        def plot_correlation(self, other_interface, lines=False, bounds=None):
            """ Plot this interface's view on the underlying product vs
            another interface's view.

            Each vulnerability will be represented as a point, with the x coordinate
            representing the likelihood that this interface will discover the
            vulnerability in the next round, and the y coordinate representing
            the likelihood that the other interface will discover it this round.

            This plot is helpful for understanding the discovery correlation that
            arises with learning.

            Parameters
            ----------
            other_interface: Interface object
               The interface object must be built upon the same underlying 'product'
               as the present interface

            lines: True/False
               If true, will overlay horizontal and vertical lines representing the
               mean discovery profile for each actor, and label them with the mean
               value.

            bounds: None or float
               If float will set the x and y limits to this value.

            TODO
            ----
            It might be a good idea to make the bounds an x/y tuple if we want to
            initialize the interfaces with different `max_area` values.

            """
            plt.plot(self.circles.area, other_interface.circles.area, '.', alpha=.1)

            if bounds == None:
                window_size = max(self.circles.area.max(), other_interface.circles.area.max())
            else:
                window_size = bounds

            if lines:
                y_mean = np.mean(other_interface.circles.area)
                x_mean = np.mean(self.circles.area)
                plt.hlines(y_mean, 0, window_size)
                plt.text(window_size, y_mean, 'mean=%f'%y_mean, ha='right', va='bottom')
                plt.vlines(x_mean, 0, window_size)
                plt.text(x_mean, window_size, 'mean=%f'%x_mean, rotation=90, ha='right', va='top')

            plt.xlim(0, window_size)
            plt.ylim(0, window_size)
            plt.box('off')
            plt.xlabel(self.name + ' likelihood of discovery', fontsize=14)
            plt.ylabel(other_interface.name + ' likelihood of discovery', fontsize=14)

        def update(self):
            """Ensures that the local 'circles' view is in sync with the underlying pts array.

            This is helpful for times when you need to update the 'offensive' team view
            to the points that remain after 'defensive' patching.
            """
            self.circles = self.circles.loc[self.circlesquare.pts.index]

        def make_circles(self, initial_dist=np.random.rand, max_area=.005):
            """Generate circles representing the 'footprint' of each vulnerability,
            or its likelihood of being discovered

            Parameters
            ----------

            initial_dist: function (0,1]
               When called, this function should return a single value from a probability
               distribution over the range (0, 1]

            max_area: float [0, infty)
               The maximum size that a circle can take on. This will be multiplied by a value
               drawn from the `initial_dist` to choose the size of a given circle.

            """
            buffer_func = lambda x: x.buffer(np.sqrt(max_area/np.pi * initial_dist()))
            self.circles = self.circlesquare.pts.apply(buffer_func)

        def seek(self, rounds, wrap=True):
            """ Look for some vulnerabilities,
            return a list telling us which of the remaining ones we've found

            Parameters
            ----------
            rounds: int
               The number of attempts to make/locations to investigate in the software

            wrap: True/False
               If circles that fall off the edge of the square should be assumed to wrap
               around to the opposite side of the square. (Implemented by searching at
               ptx +/- 1, pty +/- 1)

            Returns
            -------
            points: Pandas Series
               Index of series is unique identifiers of vulnerabilities
               Value of series is True/False for if that vulnerability is discovered this round

            """
            rounds = int(rounds) #just for error checking
            xs = np.random.rand(rounds)
            ys = np.random.rand(rounds)

            coords = []
            if wrap:
                for i in [-1, 0, 1]:
                    for j in [-1, 0, 1]:
                        coords = coords + zip(xs+i, ys+j)
            else:
                coords = zip(xs, ys)

            searchpoints = gpd.GeoSeries([Point(xy) for xy in coords])

            return self.circles.intersects(searchpoints.unary_union)

        def harden(self, rounds, wrap=True, return_patchcount=False, update=True):
            """ Perform single-actor seek and patch.

            Parameters
            ----------
            rounds: int
               The number of attempts to make/locations to investigate in the software

            wrap: Optional[True/False]
               If circles that fall off the edge of the square should be assumed to wrap
               around to the opposite side of the square. (Implemented by searching at
               ptx +/- 1, pty +/- 1)

            return_patchcount: Optional[True/False]
               If true, the function returns the number of vulns that were patched.
               To improve performance, set this to False.

            update: Optional[True/False]
               Should the actor update their own view by default after patching?

            Returns
            -------
            patch_count: int
               If `return_patchcount` is set, returns the number of vulnerabilities that
               were patched as a result of this function call.

            """
            rounds = int(rounds)
            self.circlesquare.rounds_hardened += rounds

            patch_count = self.patch(self.seek(rounds, wrap), return_patchcount)

            if update:
                self.update()

            return patch_count


        def patch(self, patch_array, return_patchcount=False):
            """ Remove a set of vulnerabilities from the underlying product.

            Parameters
            ----------
            patch_array: Pandas Series, values True/False
               Index corresponds to unique identifiers (index) of points in the
               underlying product. Value of true indicates that this particular
               vulnerability should be patched.

            return_patchcount: Optional[True/False]
               If true, the function returns the number of vulns that were patched.
               To improve performance, set this to False.

            Returns
            -------
            patch_count: int
               If `return_patchcount` is set, returns the number of vulnerabilities that
               were patched as a result of this function call.

            """
            if return_patchcount:
                count_before = self.circlesquare.count_pts()

            self.circlesquare.pts = self.circlesquare.pts[patch_array == False]

            if return_patchcount:
                count_after = self.circlesquare.count_pts()
                return count_before-count_after


        def draw(self, axes=None, circles=None, **kwargs):
            """ Draw a representation of the interface's view of the underlying
            product.

            Parameters
            ----------
            axes: a matplotlib axes object
                Can be just plt.subplot(1,1,1)

            circles: None or list of integers
                `None` plots all circles in the view
                A list of indicies plots the circles which correspond to those values

            kwargs: arguments to pass through to geopandas `plot` command
                For a list of these values, see:
                http://geopandas.org/user.html#GeoDataFrame.plot

            """
            if isinstance(circles, list):
                pltcircles = gpd.GeoDataFrame(geometry=self.circles.loc[circles])
            else:
                pltcircles = gpd.GeoDataFrame(geometry=self.circles)

            pltcircles['color'] = pltcircles.index

            if axes is None:
                axes = plt.subplot(1, 1, 1)

            pltcircles.plot(column='color', axes=axes, **kwargs)
            self.circlesquare.draw(axes, circles)


