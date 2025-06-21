from . import funcs
import numpy as np
from .image import Image
from .affine import Affine
from typing import Optional, Tuple
from numpy.typing import ArrayLike

class Optimize:
    def __init__(self, ppa: ArrayLike, ppb: ArrayLike, rotation: bool = False):
        """Affine transformation based on matching points

        OPTIMIZE(ppa, ppb) calculates the affine transform that best
        maps the points PPA (2xN matrix) to the points PPB (ditto).

        If there are three or fewer points, the result is exact, i.e.,
        a transform A for which A * PPA = PPB. In particular:

          - If there is only one pair of points, the result is a
            simple translation;
          - If there are two, a translation combined with an
            isotropic scaling and rotation;
          - If there are three, a perfectly fitting affine transform.

        If there are more than three pairs, the result is the least-
        squares solution:

            rms = sqrt(sum(||AFM * PPA - PPB||²))

        is minimized.

        In the special case of two points, the transform may be forced
        to be a pure rotation + translation (i.e., no scaling) by
        passing in True for the ROTATION flag.

        The points may also be provided as N-element lists of 2-vectors.
        """        

        if type(ppa)==list:
            ppa = np.array(ppa).T
        if type(ppb)==list:
            ppb = np.array(ppb).T

        if len(ppa.shape) !=2 or len(ppb.shape) != 2:
            raise ValueError("Inputs to Optimize should be 2xN arrays")
        if ppa.shape[0] != 2 or ppb.shape[0] != 2:
            raise ValueError("Inputs to Optimize should be 2xN arrays")
        if rotation:
            if ppa.shape[1] != 2 or ppb.shape[1] != 2:
                raise ValueError("Inputs to Optimize should be 2x2 arrays for pure rotation")

            self.afm = Affine(funcs.mirAffine(ppa, ppb))
            resi = self.afm * ppa - ppb
            err = np.sum(resi*resi, 0)
            self.rms_ = np.sqrt(np.mean(err))
            self.iworst_ = None
        else:
            afm, self.rms_, self.iworst_ = funcs.mirAffine(ppa, ppb)
        self.afm = Affine(afm)

    @property
    def affine(self):
        "The affine transform resulting from the optimization"
        return self.afm

    @property
    def rms(self):
        """The RMS error after optimization

        This is zero if the result is exact."""
        return self.rms_

    @property
    def iworst(self):
        """The worst-matched point after optimization

        This is None if three or fewer points were passed in."""
        return self.iworst_

    def __repr__(self):
        spc = "\n" + " " * len("Optimize[")
        afm = str(self.afm).split("\n")
        afm = spc.join(afm)
        return f"Optimize[{afm} {self.rms:.2f} {self.iworst}]"
