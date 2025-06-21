# __init__.py - part of spectralign

'''spectralign - Image registration tools using partially whitened spectra

This is a reimplementation by Daniel Wagenaar <daw@caltech.edu> of the
original SWiFT-IR programs (swim, mir, iscale, iavg, remod) written
by Art Wetzel <awetzel@psc.edu>.

The goal of this implementation is to make use of modern libraries like
OpenCV and numpy for efficiency and to provide a pythonic interface to
the algorithms in the hopes of facilitating integration with other tools.

This implementation does not quite (yet) implement all the functionality
of Art's programs.
'''

from .image import Image
from .affine import Affine
from .swim import Swim
from .optimize import Optimize
from . import funcs

