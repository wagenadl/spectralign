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

"""spectralign - Image registration tools using partially whitened spectra

Acknowledgments
---------------

The core logic of the `Matcher.refine` method is a reimplementation of
the “swim” program in the “SWiFT-IR” package by Art Wetzel
<awetzel@psc.edu> at the Pittsburgh Supercomputer Center. This method
was described in:

    Wetzel AW, Bakal J, Dittrich M, Hildebrand DGC, Morgan HL,
    Lichtman JW. 2016. Registering large volume serial-section
    electron microscopy image sets for neural circuit reconstruction
    using fft signal whitening. Proc. IEEE Applied Imagery Pattern
    Recognition Workshop. https://doi.org/10.1109/AIPR.2016.8010595.

That package also contained an antecedent of the `Placement.affine`
method (“mir”), although the approach to handling more than two images
in “SWiFT-IR” is very different from ours.

The core logic of the `Placement.rigid` method derives from a
conversation with Stephan Saalfeld <saalfelds@janelia.hhmi.org> at
Janelia. This method was described in:

    Ashaber M, Tomina Y, Kassraian P, Bushong EA, Kristan WB, Ellisman
    MH, Wagenaar DA, 2021. Anatomy and activity patterns in a
    multifunctional motor neuron and its surrounding circuits. Elife
    10, e61881. https://doi.org/10.7554/eLife.61881.

"""

from .image import Image
from .affine import Affine
from .swim import Matcher
from .placement import Placement
from .render import Renderer

