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


class Render:
    """Rendering of rigidly placed tiles

    Arguments:

        pos: dictionary of tile positions (from Placement.solve())
        size: either a (W, H) tuple if all tiles have same dimensions,
              or a dictionary mapping tile IDs to (W, H) tuples to
              specify dimensions on a per-tile basis
    """

    def __init__(self, pos: Dict[Tuple, Tuple],
                 size: Dict[Tuple, Tuple] or Tuple):
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
        self.image = np.zeros((self.size[1], self.size[0]))
        self.alpha = np.zeros((self.size[1], self.size[0]))

    def clear(self) -> None:
        """Reset the image in preparation for rendering another z-level"""
        self.image = np.zeros((self.size[1], self.size[0]))
        self.alpha = np.zeros((self.size[1], self.size[0]))

    def render(self, tile: Tuple, img: ArrayLike,
               apo: int = 0, hard: bool = True) -> None:
        """Render a tile into model space

        Arguments:

           tile - ID of the tile
           img - Image for the tile
           apo - radius for blending near edges
           hard - hard or soft blending

        Hard blending means that pixels where tiles overlap are taken
        from the tile where that pixel lives farthest from the edge.
        Use the `image` property to retrieve the final image.

        Soft blending means actual blending, using a cosine fall-off
        function for alpha.
        Use the `blended` property to retrieve the final image.
        """
        x0 = int(np.round(self.pos[tile][0] - self.pmin[0]))
        y0 = int(np.round(self.pos[tile][1] - self.pmin[1]))
        w = img.shape[1]
        h = img.shape[0]
        def buildalph():
            apox = np.ones(w)
            apoy = np.ones(h)
            apox[:apo] = .5 - .5*np.cos(np.pi*np.arange(apo)/apo)
            apox[-apo:] = .5 + .5*np.cos(np.pi*np.arange(apo)/apo)
            apoy[:apo] = .5 - .5*np.cos(np.pi*np.arange(apo)/apo)
            apoy[-apo:] = .5 + .5*np.cos(np.pi*np.arange(apo)/apo)
            return apox.reshape(1, -1) * apoy.reshape(-1, 1)
        if hard:
            if apo == 0:
                self.image[y0:y0+h, x0:x0+w] = img
                self.alpha[y0:y0+h, x0:x0+w] = 1
            else:
                img0 = self.image[y0:y0+h, x0:x0+w]
                alph0 = self.alpha[y0:y0+h, x0:x0+w]
                alph = buildalph()
                mask = (alph > alph0)
                img0[mask] = img[mask]
                alph0[mask] = alph[mask]
        else:
            if apo == 0:
                self.image[y0:y0+h, x0:x0+w] += img
                self.alpha[y0:y0+h, x0:x0+w] += 1
            else:
                alph = buildalph()
                self.image[y0:y0+h, x0:x0+w] += alph * img
                self.alpha[y0:y0+h, x0:x0+w] += alph

    @property
    def blended(self):
        img = self.image / (self.alpha + 1e-99)
        return img
    
