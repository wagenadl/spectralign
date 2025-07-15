# swim - part of spectralign

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

from typing import Optional, List, Tuple
import numpy.typing
type ArrayLike = numpy.typing.ArrayLike


class Swim:
    """SWIM - Calculator of alignment between two image tiles

    After construction, use the ALIGN method to perform the
    alignment.

    Optional argument WHT specifies whitening exponent.

    Optional argument RAD specifies the radius for estimating the peak.

    For convenience, ALIGN returns self, so the idiom

        dx, dy = Swim().align(img1, img2).shift()

    is a convenient way to retrieve (just) the calculated displacement.
    If you also need, e.g., the signal to noise ratio, do this instead:

        swm = Swim().align(img1, img2)
        dx, dy = swm.shift()
        snr = swm.snr()
    """
    def __init__(self, wht: float = -0.65, rad: int = 5):
        self.wht = float(wht)
        self.rad = int(rad)
        self.dxy = None
        self.sxy = None
        self.snr_ = None
        

    def align(self, sta: Image, mov: Image) -> "Swim":
        """ALIGN - Calculate alignment between image pairs      

        ALIGN(sta, mov) calculates the optimal shift between a
        pair of image tiles. Tiles are typically cut from a larger
        image using STRAIGHTWINDOW or TRANSFORMEDWINDOW. This function
        will automatically apodize the stationary image (STA) if that
        has not already been done. The results may be retrieved with
        the SHIFT property.
        
        ALIGN also calculates the estimated width of the peak and the
        overall signal-to-noise ratio SNR in the shift image. These
        may be retrieved with the PEAKWIDTH and SNR properties.

        """

        tpl = funcs.swim(sta.apodized(), mov, self.wht, self.rad)
        self.dxy = np.array([tpl[0], tpl[1]])
        self.sxy = np.array([tpl[2], tpl[3]])
        self.snr_ = tpl[4]
        return self

    @property
    def shift(self) -> np.ndarray:
        """SHIFT - Shift from preceding alignment calculation

        dx, dy = SHIFT returns the results from a preceding call to
        ALIGN.

        DX is positive if common features in image MOV are found to the
        right of the same features in image STA.

        DY is positive if common features in image MOV are found below
        the same features in image STA.

        The result is returned as a length-2 numpy array, which may be
        picked apart into scalars as indicated above.
        """
        return self.dxy

    @property
    def peakwidth(self) -> np.ndarray:
        """PEAKWIDTH - Peak width from preceding alignment calculation

        sx, sy = PEAKWIDTH returns the estimated width of the peak
        in the alignment. Extremely large or extremely small values
        may indicate failure to find a real peak.
        """
        return self.sxy
    
    @property
    def snr(self) -> float:
        """SNR - Signal-to-noise ratio from preceding alignment calculation

        r = SNR returns the overall signal-to-noise ratio SNR in the
        shift image. Small SNR values may indicate failure to find a
        real peak.

        """        
        return self.snr_

    def __repr__(self):
        if self.dxy is None:
            return f"Swim[wht={self.wht:.2f}, rad={self.rad}]"
        dxy = f"({self.dxy[0]:6.2f},{self.dxy[1]:6.2f})"
        sxy = f"({self.sxy[0]:6.2f},{self.sxy[1]:6.2f})"
        snr = f"{self.snr:6.2f}"
        return f"Swim[{dxy} ± {sxy} @ {snr}]"
    

def refine(img1: Image, img2: Image, p1: ArrayLike, p2: ArrayLike,
           siz: int or ArrayLike,
           iters: int = 1,
           tol: float = 0,
           every: bool = False,
           wht: float = -0.65,
           rad: int = 5) -> Tuple[ArrayLike, ArrayLike, Swim]:
    """Iteratively refine a pair of matching points

    Arguments:

        img1, img2: the images to be overlaid
        p1, p2: points in img1, img2 that putatively match
        siz: size of window to grab around the points
        iters: number of iterations to perform
        tol: if positive, stop early if shift less than tol

    Returns:

        p1: updated point in first image
        p2: updated point in second image
        swm: the Swim from the last iteration

    Up to `iters` iterations are performed. If at any time the
    Euclidean length of the refinement is less than `tol`, the
    process is ended early.

    The returned `p1` and `p2` are better estimates of the matching
    points in the two images.

    The returned `swm` is a Swim structure that may be used to extract
    quality estimates, including the shift, SNR, and peak width found
    in the final iteration. The structure is augmented with a `niters`
    field that records the number of iterations performed.

    """
    p1 = np.asarray(p1, float) # this copies, on purpose
    p2 = np.asarray(p2, float)
    swms = []
    for k in range(iters):
        win1 = img1.straightwindow(p1, siz)
        win2 = img2.straightwindow(p2, siz)
        swm = Swim(wht, rad).align(win1, win2)
        swm.niters = k + 1
        swms.append(swm)
        dxy = swm.shift/2
        p1 -= dxy
        p2 += dxy
        if tol and np.sum(dxy**2) < tol**2 / 4:
            break
    if every:
        return p1, p2, swms
    else:
        return p1, p2, swm



class RefinementStats:
    """The statistics returned from a `refine` call
    
    In general, a large `shift` (compared to the size of the window) is
    indicative of lack of convergence.

    A low `snr` value (relative to results from comparable images) is
    also indicative of a poor match.

    Uncommonly high `peakwidth` is also a bad sign, but a respectable
    peakwidth on its own should not be taken as proof that the match
    is OK.

    """
    
    def __init__(self):
        self.shifts: List[ArrayLike] = []
        """(x, y)-pairs of shifts in each iteration"""
        self.peakwidths: List[ArrayLike] = []
        """(sx, sy)-pairs of the size of the spectral peaks found in each iteration"""
        self.snrs: List[float] = []
        """Signal-to-noise ratios found in each iteration"""
        
    @property
    def niters(self) -> int:
        """The number of iterations performed"""
        return len(self.shifts)

    @property
    def shift(self) -> ArrayLike:
        """(x, y)-pair of shift found in final iteration"""
        return self.shifts[-1]

    @property
    def peakwidth(self) -> ArrayLike:
        """(sx, sy)-pair of the size of the spectral peaks found in final iteration"""
        return self.peakwidths[-1]

    @property
    def snr(self) -> float:
        """Signal-to-noise ratio found in final iteration"""
        return self.snrs[-1]

    @property
    def dists(self) -> List[float]:
        """Euclidean lengths of shifts found in each iteration"""
        return [np.sum(s**2)**.5 for s in self.shifts]

    @property
    def dist(self) -> float:
        """Euclidean length of shift found in final iteration"""
        return np.sum(self.shifts[-1]**2)**.5

    @property
    def totaldist(self) -> float:
        """Euclidean length of total shift across all iterations"""
        return np.sum(np.sum(self.shifts, 0)**2)**.5
        
        

class Matcher:
    """Tool to find matching points between a pair of images
    """
    def __init__(self, img1: Image, img2: Image, stats: bool = False):
        """
        Arguments:
            img1: The first image to study
            img2: The other image to study
            stats: Whether `refine` should return statistics
        """
        self.img1 = img1
        self.img2 = img2
        self.wht: Optional[float] = None
        self.rad: Optional[int] = None
        self.stats = stats
        self.setpeakradius()
        self.setprewhiten()

    def setpeakradius(self, rad: int = 5) -> None:
        """Set the peak radius parameter

        Arguments:
            rad: new radius (must be integer)

        It is rarely necessary to change this from its default value.
        """
        self.rad = rad

    def setprewhiten(self, wht: float = -0.65) -> None:
        """Set the pre-whitening argument parameter

        Arguments:
            wht: new pre-whitening value (usually between −1 and 0)

        It is rarely necessary to change this from its default value.
        """
        self.wht = float(wht)

    def refine(self,
               p1: ArrayLike,
               p2: ArrayLike,
               size: ArrayLike,
               iters: int = 1,
               tolerance: float = 0) -> Tuple[ArrayLike, ArrayLike, RefinementStats]:
        """Iteratively refine a pair of matching points

    Arguments:    
        p1: point in img1 that putatively matches...
        p2: ... this point in img2
        size: size of window to grab around the points
        iters: number of iterations to perform
        tolerance: if positive, stop early if shift less than this

    Returns:
        p1: updated point in first image
        p2: updated point in second image
        stats: optional statistics about refinement

    The returned `p1` and `p2` are better estimates of the matching
    points in the two images.

    The `stats` are only returned if enabled at construction time.

    If `size` is a single number, a square window is used. Otherwise,
    `size` must comprise a (width, height)-pair.

    Up to `iters` iterations are performed. If at any time the
    Euclidean length of the refinement is less than `tolerance`, the
    process is ended early.

    The returned `stats` may be used as an indicator of the quality of
    the results. See below.

        """      
        p1, p2, swms = refine(self.img1, self.img2, p1, p2, size,
                              iters=iters, tol=tolerance,
                              wht=self.wht, rad=self.rad, every=True)
        if self.stats:
            stats = RefinementStats()
            stats.shifts = [swm.dxy for swm in swms]
            stats.peakwidths = [swm.sxy for swm in swms]
            stats.snrs = [swm.snr for swm in swms]
            return p1, p2, stats
        else:
            return p1, p2
