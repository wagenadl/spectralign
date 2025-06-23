# __init__.py - part of spectralign

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

'''spectralign - Image registration tools using partially whitened spectra

This is a reimplementation by Daniel Wagenaar <daw@caltech.edu> of
some of the original SWiFT-IR programs (swim, mir, iscale, iavg,
remod) written by Art Wetzel <awetzel@psc.edu>, as well as some of my
own sbemalign programs.

'''

from .image import Image
from .affine import Affine
from .swim import Swim, refine
from .optimize import Optimize
from .placement import Placement
from .render import Render
from . import funcs

