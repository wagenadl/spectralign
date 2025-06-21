from . import funcs
import numpy as np
from .image import Image
from typing import Optional, Tuple
from numpy.typing import ArrayLike
import scipy.optimize

class Affine():
    """AFFINE - Affine transformations

    The constructor builds a unity transformation if given no arguments,
    or uses the optional array for source data
    """
    def __init__(self, arr: Optional[ArrayLike] = None):
        if arr is None:
            self.data = np.array([[1., 0, 0], [0, 1., 0]])
        else:
            self.data = np.array(arr)
            

    def __repr__(self):
        lst = []
        for row in self.data:
            lst.append(f"[{row[0]:8.4f} {row[1]:8.4f} | {row[2]:8.4f}]")
        return "\n".join(lst)
    
    def apply(self, img: Image, rect: Optional[ArrayLike] = None) -> Image:
        """APPLY - Apply an affine transformation to an image

        res = afm.APPLY(img) returns an image the same size of IMG
        looking up pixels in the original using affine transformation.
        res = afm.APPLY(img, rect), where rect is an (x0,y0,w,h)-tuple
        returns the given rectangle of model space.
        
        Example: if AFM is the result of MIRAFFINE(pstat, pmov), where PSTAT
        and PMOV are corresponding points on a stationary image ISTAT and a
        moving image IMOV, then afm.APPLY(imov) overlays the moving
        image on top of the stationary image.
        """
        return Image(funcs.affineImage(self.data, img.data, rect))

    
    def __imatmul__(self, afm: "Affine") -> "Affine":
        """Operator @= - Compose two affine transforms in place

        AFM1 @= AFM2 incorporates AFM2 into AFM1, such that AFM1 becomes
        the affine transform AFM1 ∘ AFM2 that applies AFM1 after AFM2.
        """
        self.data = funcs.composeAffine(self.data, afm.data)
        return self

    def __matmul__(self, afm: "Affine") -> "Affine":
        """Operator @ - Compose two affine transforms

        (AFM1 @ AFM2) returns the affine transform AFM1 ∘ AFM2
        that applies AFM1 after AFM2.
        """
        return Affine(funcs.composeAffine(self.data, afm.data))

    def __mul__(self, xy: ArrayLike) -> np.ndarray:
        """Operator * - Apply affine transformation to a point or several points

        AFM * PT, where PT is a 2-vector, returns the point after applying the
        affine transform.
        AFM * PTS, where PTS is a 2xN array, applies the affine transform to
        multiple points at once.
        """
        return funcs.applyAffine(self.data, xy)

    
    def shift(self, dxy: ArrayLike) -> "Affine":
        '''SHIFT - Add a translation to an affine transform in place

        afm.SHIFT(dx) composes the transformation x -> x + DX after
        the affine transformation AFM.'''
        self.data[0,2] += dxy[0]
        self.data[1,2] += dxy[1]
        return self

    def shifted(self, dxy: ArrayLike) -> "Affine":
        '''SHIFTED - Add a translation to an affine transform

        afm.SHIFTED(dx) composes the transformation x -> x + DX after
        the affine transformation AFM and returns the result'''
        out = Affine(self.data.copy())
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
        """INVERT - Invert affine transform in place

        afm.INVERT() inverts the transform.
        """
        self.data = funcs.invertAffine(self.data)
        return self

    def inverse(self) -> "Affine":
        """INVERSE - Inverse of an affine transform

        afm.INVERSE() returns an affine transform that is the inverse of
        the given transform.
        """
        out = Affine(self.data.copy())
        out.invert()
        return out

    def magnification(self) -> float:
        """MAGNIFICATION - Approximate linear scaling factor

        The value is exact if the transform is pure scaling combined
        with rotation and translation. 
        """
        return np.sqrt(self.data[0,0]*self.data[1,1]
                       - self.data[0,1]*self.data[1,0])

    def angle(self, refine: bool = False) -> float:
        """ANGLE - Approximate rotation angle

        The value is exact if the transform is pure scaling combined
        with rotation and translation. 
        """
        def err(p):
            phi = p[0]
            x = np.sum((self.data[:,:2]
                        - np.array([[np.cos(phi), -np.sin(phi)],
                                    [np.sin(phi), np.cos(phi)]])) ** 2)
            return x
        phi0 = np.arctan2(self.data[1,0] - self.data[0,1],
                          self.data[0,0] + self.data[1,1])
        if refine:
            p = scipy.optimize.fmin(err, [phi0])
            phi = p[0]
            return phi
        else:
            return phi0

    def translation(self) -> np.ndarray:
        """TRANSLATION - Translational part of the transform

        The value is always exact.
        """
        return self.data[:,2]

    def centerofrotation(self) -> np.ndarray:
        """CENTEROFROTATION - Approximate center of rotation

        Any affine transformation T: X -> A X + B
        can be written as T: X -> A (X - C) + C.
        This function calculates that C.

        Note that the calculation is numerically unstable near A = unity.
        """

        """Let's do the linear algebra:
        A (X - C) + C = A X + B for all X
        AX - AC + C = A X + B for all X
        AC - C = -B
        (A - 1) C = -B"""

        return np.linalg.solve(self.data[:,:2] - np.eye(2), -self.data[:,2])

    @staticmethod
    def translator(dx: float, dy: float):
        return Affine([[1., 0, dx], [0, 1., dy]])
    
    @staticmethod
    def rotator(phi: float, x0: float = 0, y0: float = 0) -> "Affine":
        c = np.cos(phi)
        s = np.sin(phi)
        return (Affine.translator(x0, y0)
                @ Affine([[c, -s, 0], [s, c, 0]])
                @ Affine.translator(-x0, -y0))

    @staticmethod
    def scaler(s: float, x0: float = 0, y0: float = 0) -> "Affine":
        return Affine([[s, 0., x0*(1-s)], [0., s, y0*(1-s)]])
    
    
