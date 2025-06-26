# renderaffine.py - part of spectralign

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
from .affine import Affine


class RenderAffine:
    """Rendering of tiles with affine placement

    Arguments:

        afms: dictionary of affine transformations (from Placement.affine())
        size: either a (W, H) tuple if all tiles have same dimensions,
              or a dictionary mapping tile IDs to (W, H) tuples to
              specify dimensions on a per-tile basis
        clip: An optional [x0, y0, w, h] bounding box in model space.
              If not given, the full extent covered by the source images
              is used. Clip may also be a single boolean True to
              automatically clip to the intersection of all images.

    Use the `render` method to render a source image into model space.

    Use the `clear` method between z-levels.

    Use the `image` or `blended` properties to retrieve results.    
    """

    def __init__(self, afms: Dict[Tuple, Tuple],
                 size: Dict[Tuple, Tuple] or Tuple,
                 clip: ArrayLike or bool = False):
        self.afms = afms
        if type(size) == dict:
            self.imgsizes = size
        else:
            self.imgsizes = {k: size for k in afms}

        if clip and type(clip) != bool:
            pmin = np.array([clip[0], clip[1]])
            pmax = pmin + np.array([clip[2], clip[3]])
        else:
            pmin = np.array([np.inf, np.inf])
            pmax = np.array([-np.inf, -np.inf])
            for t, afm in afms.items():
                w, h = self.imgsizes[t]
                p0 = afm * [0,0]
                p1 = afm * [w, 0]
                p2 = afm * [w, h]
                p3 = afm * [0, h]
                for p in [p0, p1, p2, p3]:
                    for d, x in enumerate(p):
                        pmin[d] = min(pmin[d], x)
                        pmax[d] = max(pmax[d], x)
        self.size = np.ceil(pmax - pmin).astype(int)      
        self.pmin = pmin
        self.pmax = pmax
        self.shifter = Affine.translator(-self.pmin)
        self.rect = [0, 0, self.size[0], self.size[1]]

        if clip and type(clip) == bool:
           msk = self.intersectionmask()
           x0 = np.argmax(np.max(msk,0))
           w = np.sum(np.max(msk,0))
           y0 = np.argmax(np.max(msk,1))
           h = np.sum(np.max(msk,1))
           xy0 = np.array([x0,y0])
           self.size = np.array([w,h], int)
           self.pmin += xy0
           self.pmax = self.pmin + self.size
           self.shifter = Affine.translator(-self.pmin)
           self.rect = [0, 0, self.size[0], self.size[1]]
           
        self.clear()
           

    def clear(self) -> None:
        """Reset the image in preparation for rendering another z-level"""
        self.image = np.zeros(self.size[::-1])
        self.alpha = np.zeros(self.size[::-1])

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
        h, w = img.shape
        def buildalph():
            apox = np.ones(w)
            apoy = np.ones(h)
            apox[:apo] = .5 - .5*np.cos(np.pi*np.arange(apo)/apo)
            apox[-apo:] = .5 + .5*np.cos(np.pi*np.arange(apo)/apo)
            apoy[:apo] = .5 - .5*np.cos(np.pi*np.arange(apo)/apo)
            apoy[-apo:] = .5 + .5*np.cos(np.pi*np.arange(apo)/apo)
            return apox.reshape(1, -1) * apoy.reshape(-1, 1)
        afm = self.shifter @ self.afms[tile]
        img1 = afm.apply(img, self.rect)
        if hard:
            if apo == 0:
                mask = afm.apply(np.ones(img.shape), self.rect).astype(bool)
                self.image[mask] = img1[mask]
                self.alpha = np.max((self.alpha, mask), 0)
            else:
                alph = afm.apply(buildalph(), self.rect)
                mask = alph > self.alpha
                self.img[mask] = img1[mask]
                self.alpha[mask] = alph[mask]
        else:
            if apo == 0:
                mask = afm.apply(np.ones(img.shape), self.rect).astype(bool)
                self.image[mask] += img1[mask]
                self.alpha[mask] += 1
            else:
                alph = afm.apply(buildalph(), self.rect)
                self.image += alph * img1
                self.alpha += alph

    def unionmask(self):
        img = np.zeros(self.size[::-1], bool)
        for t in self.afms:
            afm = self.shifter @ self.afms[t]
            w,h = self.imgsizes[t]
            mask = afm.apply(np.ones((h,w)), self.rect).astype(bool)
            img |= mask
        return img

    def intersectionmask(self):
        img = np.ones(self.size[::-1], bool)
        for t in self.afms:
            afm = self.shifter @ self.afms[t]
            w,h = self.imgsizes[t]
            mask = afm.apply(np.ones((h,w)), self.rect).astype(bool)
            img &= mask
        return img
        
                
    @property
    def blended(self):
        img = self.image / (self.alpha + 1e-99)
        return img
    
