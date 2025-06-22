from . import funcs
import numpy as np
from .image import Image

from typing import Optional, List, Tuple
from numpy.typing import ArrayLike


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
    
