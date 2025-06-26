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
from .affine import Affine
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
    """Tools for placing tiles in model space

    To start, add matched pairs using the `addmatch` method.
    Then, fix tiles in model space using either the `rigid`
    or `affine` methods.
    """
    def __init__(self):
        self.matches : List[Match] = []

        
    def addmatch(self, tile1: Tuple, tile2: Tuple,
                 p1: Tuple, p2: Tuple, weight : float = 1) -> None:
        """Add a pair of matching points that connect two tiles

        Matching points are commonly obtained using the `refine` function.

        Arguments

            tile1, tile2 represent the tiles to connect
        
            p1, p2 represent points within those tiles that
                   should occupy the same location in model space
        
            weight is an optional weighting factor for this connection
        """
        self.matches.append(Match(tile1, tile2, p1, p2, weight))


    def _tilemap(self):
        tilemap : Dict[Tuple, int] = {}
        N = 0
        for mtch in self.matches:
            if mtch.tile1 not in tilemap:
                tilemap[mtch.tile1] = N
                N += 1
            if mtch.tile2 not in tilemap:
                tilemap[mtch.tile2] = N
                N += 1
        return tilemap

    
    def rigid(self, fix=None) -> Dict[Tuple, Tuple]:
        """Rigid solution with only translation

        This places all tiles in model space and returns a dictionary
        that maps tile IDs to (x, y) positions for the tile.

        The inherent ambiguity in overall position is resolved by
        enforcing the average of all tile positions to lie at (0, 0).

        As an alternative, the `fix` parameter may be used to
        specify that one particular tile lies at (0, 0).
        """
        tilemap = self._tilemap()
        N = len(tilemap)
        txy = []
        for dim in range(2):
            A = np.zeros((N+1,N+1))
            b = np.zeros(N+1)
            for mtch in self.matches:
                # A Match of tiles t1, t2 connecting points x1 to x2
                # with weight w gives rise to a term
                # E = w ((X_t1 + x1) - (X_t2 + x2))²
                # See also solveq5ridigtile.py and matchpointsq5.py
                # in sbemalign.
                # Rather than stabilizing terms E₀ = ε X²_t, this
                # version uses Lagrange multipliers to force Σ_t X_t = 0.
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
            A[:-1,-1] = 1
            A[-1,:-1] = 1

            if fix is not None:
                tfix = tilemap[fix]
                i0 = tfix
                i1 = i0 + 1
                A = np.delete(A, range(i0, i1), 0)
                A = np.delete(A, range(i0, i1), 1)
                b = np.delete(b, range(i0, i1))
                A = A[:-1,:-1]
                b = b[:-1]
            
            X = np.linalg.solve(A, b)[:-1]

            if fix is not None:
                X = np.concatenate((X[:i0], [0], X[i1-1:]))
            txy.append(X)
            
        rigidpos = {}
        for tid, t in tilemap.items():
            rigidpos[tid] = (txy[0][t], txy[1][t])
        return rigidpos
    
    
    def affine(self, fix=None) -> Dict[Tuple, Affine]:
        """Rigid solution with affine transformations

        This places all tiles in model space and returns a dictionary
        that maps tile IDs to affine transformations for the tile.

        The inherent ambiguities in scale and overall position are
        resolved by enforcing the average of all tile positions to lie
        at (0, 0) and that the average transformation matrix is
        unity. This works best if the expected transformations are
        distributed near unity to begin with. For instance, rotations
        greater than 90° generally do not lead to good results,
        because averaging matrices is not reasonable in that case. (A
        theoretically better constraint might be to demand that the
        determinants of the transforms average to one, but that does
        not lead to a linear Lagrange multiplier.)

        As an alternative, you can fix one of the transforms to unity
        using the `fix` parameter. This is only recommended if every
        tile is strongly connected to the fixed tile with matching
        points. Otherwise, the solution will bias toward reduced scale
        for tiles farther from the fixed one.

        """

        tilemap = self._tilemap()
        N = len(tilemap)
        M = np.zeros((6*N + 6, 6*N + 6))
        b = np.zeros((6*N + 6))

        def Aindex(t, i, j):
            return t*6 + 2*i + j
        def Xindex(t, i):
            return t*6 + 4 + i
        def LLindex(i, j):
            return N*6 + 2*i + j
        def Lindex(i):
            return N*6 + 4 + i

        def addmatch(m, m_, p, p_, w=1):
            # These are the dE/dA terms in A and X
            for n in range(2):
                for i in range(2):
                    for j in range(2):
                        M[Aindex(m,n,i), Aindex(m,n,j)] += w*p[i] * p[j]
                        M[Aindex(m,n,i), Aindex(m_,n,j)] -= w*p[i] * p_[j]
                    M[Aindex(m,n,i), Xindex(m, n)] += w * p[i]
                    M[Aindex(m,n,i), Xindex(m_, n)] -= w * p[i]

            # These are the dE/dX terms in A and X
            for n in range(2):
                for j in range(2):
                    M[Xindex(m,n), Aindex(m,n,j)] += w * p[j]
                    M[Xindex(m,n), Aindex(m_,n,j)] -= w * p_[j]
                M[Xindex(m,n), Xindex(m,n)] += w
                M[Xindex(m,n), Xindex(m_,n)] -= w

        def addLterms():
            # These are the dE/dA terms in LL
            for m in range(N):
                for i in range(2):
                    for j in range(2):
                        M[Aindex(m,i,j), LLindex(i,j)] += 1

            # These are the dE/dX terms in L
            for m in range(N):
                for i in range(2):
                    M[Xindex(m,i), Lindex(i)] += 1


            # These are the dE/dLL terms in A
            for m in range(N):
                for i in range(2):
                    for j in range(2):
                        M[LLindex(i,j), Aindex(m,i,j)] += 1

            # These are the constant dE/dLL terms
            for i in range(2):
                b[LLindex(i,i)] += N

            # These are the dE/dL terms in X
            for m in range(N):
                for i in range(2):
                    M[Lindex(i), Xindex(m,i)] += 1

            # There are no constant dE/dL terms
            
        for mtch in self.matches:
            t1 = tilemap[mtch.tile1]
            t2 = tilemap[mtch.tile2]
            addmatch(t1, t2, mtch.p1, mtch.p2, mtch.weight)
            addmatch(t2, t1, mtch.p2, mtch.p1, mtch.weight)
        addLterms()
        
        if fix is not None:
            tfix = tilemap[fix]
            i0 = 6*tfix
            i1 = i0 + 6
            coefs = M[:, i0:i1]
            consts = np.sum(coefs * [1,0,0,1,0,0], 1)
            b -= consts
            M = np.delete(M, range(i0, i1), 0)
            M = np.delete(M, range(i0, i1), 1)
            b = np.delete(b, range(i0, i1))
            M = M[:-6,:-6]
            b = b[:-6]
        
        AX = np.linalg.solve(M, b)

        if fix is not None:
            AX = np.concatenate((AX[:i0], [1,0,0,1,0,0], AX[i1-6:]))

        AA = np.zeros((N,2,2))
        for m in range(N):
            for i in range(2):
                for j in range(2):
                    AA[m,i,j] = AX[Aindex(m,i,j)]

        XX = np.zeros((N,2))
        for m in range(N):
            for i in range(2):
                 XX[m,i] = AX[Xindex(m,i)]
   
        afms = {}
        for tid, t in tilemap.items():
            afms[tid] = Affine(np.hstack((AA[t], XX[t].reshape(2,1))))
        return afms
 

    # An elasticsolve() method should be programmed next.
    # This results in a map of all the points in the data set
    # to image space.
    # From that, a meshgrid can be constructed.
