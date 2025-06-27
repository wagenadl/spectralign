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
    def __init__(self, wht: float = -0.65, rad: int = 5):
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


class Matcher:
    """Tool to find matching points between a pair of images

    Arguments:

        img1, img2: The images to study

    Use the `refine` method to identify matching points in the images.
    """
    def __init__(self, img1: Image, img2: Image):
        self.img1 = img1
        self.img2 = img2
        self.wht = None
        self.rad = None
        self.shifts = []
        self.peakwidths = []
        self.snrs = []
        self.setpeakradius()
        self.setprewhiten()

    def setpeakradius(self, rad: int = 5):
        self.rad = rad

    def setprewhiten(self, wht: float = -0.65):
        self.wht = float(wht)

    def refine(self,
               p1: ArrayLike, p2: ArrayLike,
               size: int or ArrayLike,
               iters : int = 1,
               tolerance : float = 0) -> Tuple[ArrayLike, ArrayLike]:
        """Iteratively refine a pair of matching points

    Arguments:

        p1, p2: points in img1, img2 that putatively match
        siz: size of window to grab around the points
        iters: number of iterations to perform
        tol: if positive, stop early if shift less than tol

    Returns:

        p1: updated point in first image
        p2: updated point in second image

    The returned `p1` and `p2` are better estimates of the matching
    points in the two images.

    Up to `iters` iterations are performed. If at any time the
    Euclidean length of the refinement is less than `tolerance`, the
    process is ended early.

    After `refine` returns, several properties are set on the Matcher:

       shifts: (x, y) pairs of shifts in each iteration
    
       peakwidths: (sx, sy) pairs of the size of the spectral peaks
                   found in each iteration

       snrs: signal-to-noise ratios found in each iteration

       niters: the number of iterations performed

    In addition, `shift`, `peakwidth`, and `snr` may be used to obtain
    information from the final iteration.
    """      
        p1, p2, swms = refine(self.img1, self.img2, p1, p2, size,
                              iters=iters, tol=tolerance,
                              wht=self.wht, rad=self.rad, every=True)
        self.shifts = [swm.dxy for swm in swms]
        self.peakwidths = [swm.sxy for swm in swms]
        self.snrs = [swm.snr for swm in swms]
        return p1, p2

    @property
    def niters(self):
        return len(self.shifts)

    @property
    def shift(self):
        return self.shifts[-1]

    @property
    def peakwidth(self):
        return self.peakwidths[-1]

    @property
    def snr(self):
        return self.snrs[-1]
    
