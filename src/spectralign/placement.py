# placement.py - part of spectralign

## Copyright (C) 2025  Daniel A. Wagenaar
## 
## This program is free software: you can redistribute it and/or
## modify it under the terms of the GNU General Public License as
## published by the Free Software Foundation, either version 3 of the
## License, or (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
## General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <https://www.gnu.org/licenses/>.


from . import funcs
import numpy as np
from typing import Optional, Tuple, List, Dict
import numpy.typing
type ArrayLike = numpy.typing.ArrayLike
from collections import namedtuple

Match = namedtuple("Match", ("tile1", "tile2", "p1", "p2", "weight"))
"""A Match represents a pair of points in two different tiles that
should occupy the same location in model space.

    tile_k: any tuple identifying the tiles, e.g., (r, m, s) form
    p_k: (x, y) coordinate pairs
    weight: weighting factor for this connection
"""

class Placement:
    """
    """
    def __init__(self):
        self.matches : List[Match] = []
        self.rigidpos: Dict[Tuple, Tuple] = {}
        self.epsilon = 1e-6

    def addmatch(self, tile1: Tuple, tile2: Tuple,
                 p1: Tuple, p2: Tuple, weight : float = 1) -> None:
        """Add a pair of matching points that connect two tiles

        Arguments

            tile1, tile2 represent the tiles to connect
        
            p1, p2 represent points within those tiles that
                   should occupy the same location in model space
        
            weight is an optional weighting factor for this connection
        """
        self.matches.append(Match(tile1, tile2, p1, p2, weight))

        
    def solve(self, verbose=False) -> Dict[Tuple, Tuple]:
        """
        """
        if verbose:
            print("Building matrix")
        tilemap : Dict[Tuple, int] = {}
        N = 0
        for mtch in self.matches:
            if mtch.tile1 not in tilemap:
                tilemap[mtch.tile1] = N
                N += 1
            if mtch.tile2 not in tilemap:
                tilemap[mtch.tile2] = N
                N += 1
        txy = np.zeros((2, N))
        for dim in range(2):
            if verbose:
                print(f"Solving {dim+1}/2")
            A = np.eye(N) * self.epsilon
            b = np.zeros(N)
            for mtch in self.matches:
                # A Match of tiles t1, t2 connecting points x1 to x2
                # with weight w gives rise to a term
                # E = w ((t1x + x1) - (t2x + x2))²
                # See also solveq5ridigtile.py and matchpointsq5.py
                # in sbemalign.
                t1 = tilemap[mtch.tile1]
                t2 = tilemap[mtch.tile2]
                x1 = mtch.p1[dim]
                x2 = mtch.p2[dim]
                w = mtch.weight
                A[t1, t1] += w
                A[t2, t2] += w
                A[t1, t2] -= w
                A[t2, t1] -= w
                b[t1] += w * (x2 - x1)
                b[t2] -= w * (x2 - x1)
                txy[dim] = np.linalg.solve(A, b)
        for tid, t in tilemap.items():
            self.rigidpos[tid] = (txy[0,t], txy[1,t])
        return self.rigidpos


    # An elasticsolve() method should be programmed next.
    # This results in a map of all the points in the data set
    # to image space.
    # From that, a meshgrid can be constructed.
