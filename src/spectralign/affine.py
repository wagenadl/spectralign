# affine.py - part of spectralign

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
from .image import Image
from typing import Optional, Tuple
import numpy.typing
type ArrayLike = numpy.typing.ArrayLike
import scipy.optimize

class Affine(np.ndarray):
    """Affine transformation

    The constructor builds a unity transformation if given no arguments,
    or uses the optional array for source data
    """
    def __new__(cls, arr: Optional[ArrayLike] = None):
        if arr is None:
            obj = np.asarray([[1., 0., 0.], [0., 1., 0.]]).view(cls)
        else:
            obj = np.asarray(arr, float).view(cls)
        return obj
            
    def __array_finalize__(self, obj):
        return

    def __repr__(self):
        if len(self.shape)==2:
            lst = []
            for row in self:
                lst.append(f"[{row[0]:8.4f} {row[1]:8.4f} | {row[2]:8.4f}]")
            return "\n".join(lst)
        elif len(self.shape)==0:
            return self.view(type=np.ndarray).flatten()[0].__repr__()
        else:
            return self.view(type=np.ndarray).__repr__()

    def __str__(self):
        return self.__repr__()
        
    
    def apply(self, img: Image, rect: Optional[ArrayLike] = None) -> Image:
        """Apply an affine transformation to an image

        Arguments:
            img: an image to transform
            rect: optional rectangle of model space

        Returns:
            the transformed image

        `res = afm.apply(img)` returns an image the same size of `img`
        looking up pixels in the original using affine transformation.
        
        `res = afm.apply(img, rect)`, where rect is an (x0,y0,w,h)-tuple
        returns the given rectangle of model space.
        
        Note that `afm` must map from image space to model space, not
        the other way around.

        """
        return Image(funcs.affineImage(self.inverse(), img, rect))

    
    def __imatmul__(self, afm: "Affine") -> "Affine":
        """Compose two affine transformations in place

        `afm1 @= afm2` incorporates `afm2` into `afm1`, such that
        `afm1` becomes the affine transformation *afm*₁ ∘ *afm*₂ that
        applies *afm*₁ after *afm*₂.

        """
        self[:] = funcs.composeAffine(self, afm)[:]
        return self

    def __matmul__(self, afm: "Affine") -> "Affine":
        """Compose two affine transformations

        `(afm1 @ afm2)` returns the affine transformation *afm*₁ ∘ *afm*₂
        that applies *afm*₁ after *afm*₂.
        """
        return Affine(funcs.composeAffine(self, afm))

    def __mul__(self, xy: ArrayLike) -> np.ndarray:
        """Apply affine transformation to a point or several points

        `afm * pt`, where `pt` is a 2-vector, returns the point after applying the
        affine transformation.
        
        `afm * pts`, where `pts` is a 2×*N* array, applies the affine
        transformation to multiple points at once.

        """
        return funcs.applyAffine(self, xy)

    
    def shift(self, dxy: ArrayLike) -> "Affine":
        '''Add a translation to an affine transformation in place

        Arguments:
            dxy: (x,y)-pair specifying the shift to be applied

        Returns:
            self

        `afm.shift(dx)` composes the transformation *x* → *x* + *dx* after
        the affine transformation *afm*.'''
        self[0,2] += dxy[0]
        self[1,2] += dxy[1]
        return self

    def shifted(self, dxy: ArrayLike) -> "Affine":
        """Add a translation to an affine transformation

        Arguments:
            dxy: (x,y)-pair specifying the shift to be applied

        Returns:
            the new affine transformation


        `afm.shifted(dx)` composes the transformation *x* → *x* + *dx*
        after the affine transformation *afm* and returns the result.

        """
        
        out = self.copy()
        return out.shift(dxy)
    

#    def rotate(self, phi: float, xy0: Optional[ArrayLike] = None) -> "Affine":
#        if xy0 is not None:
#            self.shift([-xy0[0], -xy0[1]])
#        self.data = funcs.composeAffine(self.data,
#                                        np.array([[np.cos(phi), -np.sin(phi), 0],
#                                                  [np.sin(phi), np.cos(phi), 0]]))
#        if xy0 is not None:
#            self.shift(xy0)
#        return self
#
#    def rotated(self, phi: float, xy0: Optional[ArrayLike] = None) -> "Affine":
#        out = Affine(self.data.copy())
#        return out.rotate(phi, xy0)
#
#
#    def scale(self, s: float, xy0: Optional[ArrayLike] = None) -> "Affine":
#        if xy0 is not None:
#            self.shift([-xy0[0], -xy0[1]])
#        self.data[:,:2] @= np.array([[s, 0], [0, s]])
#        if xy0 is not None:
#            self.shift(xy0)
#        return self
#
#    def scaled(self, s: float, xy0: Optional[ArrayLike] = None) -> "Affine":
#        out = Affine(self.data.copy())
#        return out.scale(s, xy0)


    def invert(self) -> "Affine":
        """Invert affine transformation in place

        `afm.invert()` inverts the transformation.
        """
        self[:] = funcs.invertAffine(self)[:]
        return self

    def inverse(self) -> "Affine":
        """Inverse of an affine transformation

        Returns:
            the resulting transformation

        afm.INVERSE() returns an affine transformation that is the inverse of
        the given transformation.
        """
        out = Affine(self.copy())
        out.invert()
        return out

    def magnification(self) -> float:
        """Approximate linear scaling factor

        Returns:
            the scaling factor

        We calculate this as the square root of the determinant of the
        linear part of the transformation.

        The value is exact if the transformation is pure scaling combined
        with rotation and translation.

        """
        return np.sqrt(self[0,0]*self[1,1]
                       - self[0,1]*self[1,0])

    def angle(self, refine: bool = False) -> float:
        """Approximate rotation angle

        Arguments:
            refine: if true, use scipy to optimize

        Returns:
            angle in radians

        We calculate the arctangent of the ratio between off-diagonal
        and diagonal elements of the linear part of the
        transformation. If `refine` is true, we then further optimize
        using scipy.

        The value is exact if the transformation is pure scaling combined
        with rotation and translation.

        """
        def err(p):
            phi = p[0]
            x = np.sum((self[:,:2]
                        - np.array([[np.cos(phi), -np.sin(phi)],
                                    [np.sin(phi), np.cos(phi)]])) ** 2)
            return x
        phi0 = np.arctan2(self[1,0] - self[0,1],
                          self[0,0] + self[1,1])
        if refine:
            p = scipy.optimize.fmin(err, [phi0])
            phi = p[0]
            return phi
        else:
            return phi0

    def translation(self) -> np.ndarray:
        """Translational part of the transformation

        Returns:
            the shift as a (dx, dy)-pair.

        The value is always exact.
        """
        return self[:,2].view(type=np.ndarray)

    def centerofrotation(self) -> np.ndarray:
        """Approximate center of rotation and scaling

        Returns:
            center of rotation and scaling

        Any affine transformation *T*: *x* → *A* *x* + *b*
        can be written as *T*: *x* → *A* (*x* − *c*) + *c*.
        This function calculates that *c*.

        Note that the calculation is numerically unstable near *A* = 𝟙.
        """

        """Let's do the linear algebra:
        A (X - C) + C = A X + B for all X
        AX - AC + C = A X + B for all X
        AC - C = -B
        (A - 1) C = -B"""

        return np.linalg.solve(self[:,:2] - np.eye(2), -self[:,2]).view(type=np.ndarray)

    @staticmethod
    def translator(dxy: ArrayLike) -> "Affine":
        """An affine transformation that represents a translation

        Arguments:
            dxy: a (dx, dy) pair

        Returns:
            the constructed affine transformation
        """
        dx, dy = dxy
        return Affine([[1., 0, dx], [0, 1., dy]])
    
    @staticmethod
    def rotator(phi: float, xy0: ArrayLike = [0,0]) -> "Affine":
        """An affine transformation that represents a rotation

        Arguments:
            phi: the rotation in radians
            xy0: the point around which to rotate

        Returns:
            the constructed affine transformation

        If `xy0` is not given, the rotation is around the origin.
        """
        c = np.cos(phi)
        s = np.sin(phi)
        x0, y0 = xy0
        return (Affine.translator([x0, y0])
                @ Affine([[c, -s, 0], [s, c, 0]])
                @ Affine.translator([-x0, -y0]))

    @staticmethod
    def scaler(s: float, xy0: ArrayLike = [0,0]) -> "Affine":
        """An affine transformation that represents a linear scaling

        Arguments:
            s: the scale factor
            xy0: the point around which to scale

        Returns:
            the constructed affine transformation

        If `xy0` is not given, the scaling is around the origin.

        The scaling is always isotropic, that is, we do not support
        different scale factors for x and y.
        """
        x0, y0 = xy0
        return Affine([[s, 0., x0*(1-s)], [0., s, y0*(1-s)]])
    
    
