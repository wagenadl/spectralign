# render.py - part of spectralign

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


import numpy as np
from typing import Optional, Tuple, List, Dict
import numpy.typing
type ArrayLike = numpy.typing.ArrayLike
from collections import namedtuple
from .image import Image
from scipy.signal import butter, filtfilt


class RenderRigid:
    """Rendering of rigidly placed tiles

    Arguments:

        pos: dictionary of tile positions (from Placement.rigid())
        size: either a (W, H) tuple if all tiles have same dimensions,
              or a dictionary mapping tile IDs to (W, H) tuples to
              specify dimensions on a per-tile basis

    Use the `render` method to render a source image into model space.

    Use the `clear` method between z-levels.

    Use the `image` or `blended` properties to retrieve results.
    """

    def __init__(self, pos: Dict[Tuple, Tuple],
                 size: Dict[Tuple, Tuple] or Tuple,
                 blend=10):
        self.blend = blend
        if type(size) != dict:
            size = {k: size for k in pos}
        pmin = [np.inf, np.inf]
        pmax = [-np.inf, -np.inf]
        for t, p in pos.items():
            for d, x in enumerate(p):
                pmin[d] = min(pmin[d], x)
                pmax[d] = max(pmax[d], x + size[t][d])
        self.size = (int(np.ceil(pmax[0] - pmin[0])),
                     int(np.ceil(pmax[1] - pmin[1]))) # x, y
        self.pmin = pmin
        self.pmax = pmax
        self.pos = pos
        self.clear()

    def clear(self) -> None:
        """Reset the image in preparation for rendering another z-level"""
        self.image = Image(np.zeros((self.size[1], self.size[0])))
        self.support = Image(np.zeros((self.size[1], self.size[0])))

    def render(self, tile: Tuple, img: ArrayLike) -> None:
        """Render a tile into model space

        Arguments:

           tile - ID of the tile
           img - Image for the tile
        """
        x0 = int(np.round(self.pos[tile][0] - self.pmin[0]))
        y0 = int(np.round(self.pos[tile][1] - self.pmin[1]))
        w = img.shape[1]
        h = img.shape[0]
        def buildalph():
            apox = np.ones(w)
            apoy = np.ones(h)
            w2 = min(w//2, h//2)
            wc = np.arange(w2)/w2
            apox[:w2] = wc
            apox[-w2:] = wc[::-1]
            apoy[:w2] = wc
            apoy[-w2:] = wc[::-1]
            return np.min([apox.reshape(1,-1).repeat(h,0),
                           apoy.reshape(-1,1).repeat(w,1)], 0)
        sup1 = np.zeros((self.size[1], self.size[0]))
        img1 = np.zeros((self.size[1], self.size[0]))
        sup1[y0:y0+h, x0:x0+w] = buildalph()
        img1[y0:y0+h, x0:x0+w] = img
        mask = sup1 > self.support
        if self.blend > 1:
            b,a = butter(1, 1/self.blend)
            mask = filtfilt(b, a, mask, axis=0)
            mask = filtfilt(b, a, mask, axis=1)
        mask[sup1 == 0] = 0
        mask[self.support == 0] = 1
        self.image = (1-mask)*self.image + mask * img1
        self.support = np.max([self.support, sup1], 0)
    
