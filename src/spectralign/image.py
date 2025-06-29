# image.py - part of spectralign

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
import cv2
from typing import Optional, List
import numpy.typing
type ArrayLike = numpy.typing.ArrayLike

try:
    from numba import jit
except ImportError:
    # Create fake jit
    def jit(nopython=True):
        def decor(func):
            def wrap(*args, **kwargs):
                return func(*args, **kwargs)
            return wrap
        return decor

# Following function from https://gist.github.com/bzamecnik/33e10b13aae34358c16d1b6c69e89b01
@jit(nopython=True)
def floyd_steinberg(image):
    # image: np.array of shape (height, width), dtype=float, 0.0-1.0
    # works in-place!
    h, w = image.shape
    for y in range(h):
        for x in range(w):
            old = image[y, x]
            new = np.round(old)
            image[y, x] = new
            error = old - new
            # precomputing the constants helps
            if x + 1 < w:
                image[y, x + 1] += error * 0.4375 # right, 7 / 16
            if (y + 1 < h) and (x + 1 < w):
                image[y + 1, x + 1] += error * 0.0625 # right, down, 1 / 16
            if y + 1 < h:
                image[y + 1, x] += error * 0.3125 # down, 5 / 16
            if (x - 1 >= 0) and (y + 1 < h): 
                image[y + 1, x - 1] += error * 0.1875 # left, down, 3 / 16
    return image
    

@jit(nopython=True)
def braillepix(img):
    p = 0
    k = 1
    for dx in range(2):
        for dy in range(3):
            if img[dy,dx]:
                p += k
            k *= 2
    for dx in range(2):
        for dy in [3]:
            if img[dy,dx]:
                p += k
            k *= 2

    return chr(0x2800 + p)
    
    

class Image(np.ndarray):
    """A representation of an image as a 2D array

    Images can be constructed in several ways:

    * from numpy arrays using

          img = Image(array)

    * loaded from an image file using

          img = Image.load(filename)

    * pulled from a video file using

          vidcap = cv2.VideoCapture(inputfilename)
          ...
          img = Image.readframe(vidcap)

      In this case, a RuntimeError is raised at end-of-file.

    Our native data format is np.float32. For convenience, np.uint8 or
    np.uint16 is also accepted. The intensity of such images is scaled
    by a factor of 255 or 65535, respectively.
    
    We do not keep color information. If YxXxC images are provided, the
    color channel is averaged away with equal weights for each channel.

    An Image is just a numpy array with the following additional methods:

        stretch - Strech contrast of an image in place
        stretched - Contrast-stretched copy of an image
        scaled - Geometrically scale an image down by integer factor
        apodize - Multiply a windowing function into an image in place
        apodized - Apodized copy of an image
        ascii - ASCII-art representation of an image
        save - Save an image to a file
    """

    apo = None
    apo0 = None

    @staticmethod
    def load(path: str):
        data = cv2.imread(path, cv2.IMREAD_ANYDEPTH + cv2.IMREAD_GRAYSCALE)
        return Image(data)

    @staticmethod
    def readframe(vidcap: cv2.VideoCapture):
        suc, data = vidcap.read()
        if not suc:
            raise RuntimeError("End of stream")
        return Image(data)
    
    def __new__(cls, data: ArrayLike):
        if type(data)==str:
            fn = data
            data = cv2.imread(fn,
                              cv2.IMREAD_ANYDEPTH + cv2.IMREAD_GRAYSCALE)
            if data is None:
                raise FileNotFoundError(fn)
        obj = np.asarray(data).view(cls)
        if obj.dtype == np.uint8:
            scl = 255
        elif obj.dtype == np.uint16:
            scl = 65535
        elif obj.dtype == np.uint32:
            scl = 2**32 - 1
        else:
            scl = 1
        if len(obj.shape) == 3:
            obj = obj.mean(-1).astype(np.float32)
        if len(obj.shape) != 2:
            raise ValueError("Data must be two-dimensional")
            
        if obj.dtype != np.float32:
            obj = obj.astype(np.float32)
        if scl != 1:
            obj /= scl
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.apo = getattr(obj, "apo", False)


    #def copy(self):
    #    img = np.sarray(self).copy())
    #    return img
        
    def stretch(self, percent: float = 0.1) -> "Image":
        """STRETCH - Stretch contrast of an image in place
        STRETCH(perc) stretches the contrast of an image in-place.
        PERC specifies what percentage of pixels become white or black.
        """
        N = self.size
        ilo = int(.01*stretch*N)
        ihi = int((1-.01*stretch)*N)
        vlo = np.partition(self.flatten(), ilo)[ilo]
        vhi = np.partition(self.flatten(), ihi)[ihi]
        nrm = np.array([1/(vhi-vlo)], dtype=np.float32)
        self -= vlo
        self *= nrm
        self[self < 0] = 0
        self[self > 1] = 1
        return self
        
    def stretched(self, percent: float = 0.1) -> "Image":
        """STRETCHED - Contrast-stretched copy of an image
        img.STRETCHED(perc) returns a contrast-stretched copy of an image.
        PERC specifies what percentage of pixels become white or black.
        """
        img = self.copy()
        img.stretch(percent)
        return img

    
    def scaled(self, fac: int = 2) -> "Image":
        '''SCALED - Geometrically scale an image down by integer factor
        im1 = img.SCALED(fac) returns a version of the image IMG scaled
        down by the given factor in both dimensions. FAC is optional and 
        defaults to 2. FAC must be integer.
        Before scaling, the image is trimmed on the right and bottom so
        that the width and height are a multiple of FAC.
        Pixels in the output image are the average of all underlying pixels
        in the input image.'''
        Y, X = self.shape
        Y1 = Y//fac
        X1 = X//fac
        return Image(self[:Y1*fac, :X1*fac]
                     .reshape(Y1, fac, X1, fac)
                     .mean((1, 3)))
        
    
    def apodize(self, gray: Optional[float] = None, force: bool = False) -> "Image":
        '''APODIZE - Multiply a windowing function into an image in place
        
        img.APODIZE() multiplies the outer 1/4 of the image with a cosine 
        fadeout to gray (defined as the mean of the image, unless specifically
        given as optional argument).

        The call has no effect if the image has already been apodized,
        unless FORCE is given as True.

        This function is not threadsafe, as it stores the most recent
        apodization kernel for reuse.

        '''
        if self.apo and not force:
            return self
        funcs.apodize(self, gray, inplace=True)
        self.apo = True
        return self

    def apodized(self, gray: Optional[float] = None, force: bool = False) -> "Image":
        """APODIZED - Apodized copy of an image
        
        im1 = img.APODIZED() copies the image and then applies the
        APODIZE function.  If the image is already apodized, the
        original image is returned without copying.

        """
        if self.apo and not force:
            return self
        img = self.copy()
        img.apodize(gray, force)
        return img

    
    def ascii(self, height: int = 22) -> List[str]:
        """ASCII - ASCII-art copy of an image
        img.ASCII() returns a copy of the image converted to ascii art at low
        resolution. Optional parameter HEIGHT specifies the desired height
        of the output. Intended to get a quick overview for use in
        terminals.
        """
        chrs = """·-+⊞▤▦▩■"""
        M = len(chrs)
        Y, X = self.shape
        scl = int(np.round(Y/height+.99))
        w1 = X // scl
        h1 = Y // scl
        img = self[:h1*scl,:w1*scl].reshape(h1, scl, w1, scl)
        img = (img.mean(-1).mean(-2).view(np.ndarray)*M).astype(int)
        img[img<0] = 0
        img[img>=M] = M - 1
        lines = []
        for row in img:
            lines.append("".join([chrs[pix] for pix in row]))
        return lines

    def braille(self, height: int = 88) -> List[str]:
        Y, X = self.shape
        scl = int(np.round(Y/height+.99))
        w1 = X // scl
        h1 = Y // scl
        img = self[:h1*scl,:w1*scl].reshape(h1, scl, w1, scl)
        img = img.mean(-1).mean(-2)
        img[img<0] = 0
        img[img>1] = 1
        img = floyd_steinberg(img)
        out = []
        for y in range(0, h1-3, 4):
            line = ''
            for x in range(0, w1-1, 2):
                line += braillepix(img[y:y+4, x:x+2])
            out.append(line)
        return out

    def __repr__(self):
        if len(self.shape)==2:
            lines = self.braille()
            while len(lines) < 6:
                lines.append('')
            lines[-6] += f"   Width:  {self.shape[1]} px"
            lines[-5] += f"   Height: {self.shape[0]} px"
            lines[-4] += f"   Min:  {self.min():.3f}"
            lines[-3] += f"   Max:  {self.max():.3f}"
            lines[-2] += f"   Mean: {self.mean():.3f}"
            lines[-1] += f"   SD:   {self.std():.3f}"
            return "\n".join(lines)
        elif len(self.shape)==0:
            return self.view(np.ndarray).flatten()[0].__repr__()
        else:
            return self.view(np.ndarray).__repr__()

    def __str__(self):
        return self.__repr__()
    
    def save(self, path: str, qual: int = None) -> None:
        '''SAVE - Save an image to a file
        img.SAVE(path) saves the image IMG to the file named PATH.
        Optional argument QUAL specifies jpeg quality as a number between
        0 and 100, and must only be given if PATH ends in ".jpg".'''
        img = (self.view(np.ndarray) * 255.99).astype(np.uint8)
        funcs.saveImage(img, path, qual)


    def straightwindow(self,
                       xy: ArrayLike = None,
                       siz: int = 512,
                       border: float = None) -> "Image":
        '''STRAIGHTWINDOW - Extract a window from an image
        win = img.STRAIGHTWINDOW(xy, siz) extracts a window of
        size SIZ, centered on XY, from the given image. XY do not have to
        be integer-valued.
        SIZ may be a scalar or a pair of numbers (width, height).
        SIZ defaults to 512.
        If XY is not given, the window is taken from the center of the image.
        It is OK if the window does not fit fully inside of the image. In
        that case, undefined pixels are given the value of the optional
        BORDER argument, or of the average of the image if not specified.'''    
        return Image(funcs.extractStraightWindow(self, xy, siz, border))

    
    def roi(self, rect: ArrayLike) -> "Image":
        '''ROI - Extract rectangular window from an image
        win = img.ROI((x0,y0,w,h)) extracts a rectangular window
        from an image.
        X0, Y0, W, and H must be integers. ROIs must fit inside the image.
        See also EXTRACTSTRAIGHTWINDOW.'''
        return Image(funcs.roi(self, rect))
    
        
    def transformedwindow(self, xy: ArrayLike = None,
                          tfm: ArrayLike = None,
                          siz: int = 512) -> "Image":
        '''TRANSFORMEDWINDOW - Extract a window with transform
        win = img.TRANSFORMEDWINDOW(xy, tfm, siz) extracts a window of
        size SIZ, centered on XY, from the given image.
        TFM must be a 2x2 transformation matrix. It is internally modified with
        a translation T such that the center point XY is not moved by the 
        combination of translation and transformation.
        SIZ may be a scalar or a pair of numbers (width, height). 
        SIZ defaults to 512.
        If TFM is not given, an identity matrix is used.
        If XY is not given, the window is taken from the center of the image.
        To transform an entire image, AFFINE.APPLY may be easier to use.'''
        return Image(funcs.extractTransformedWindow(self, xy, tfm, siz))

    def rmsdelta(self, img2: "Image", margin: float = 0.05) -> float:
        """RMSDELTA - RMS difference between images

        rms = img1.RMSDELTA(img2) calculates the RMS difference
        between a pair of images, ignoring edge pixels.
        
        By default, each of the four edges is 5% of the corresponding
        image dimension. The MARGIN parameter overrides the default.
        """
        return np.mean(self.delta(img2, margin)**2)**0.5
    
    def delta(self, img2: "Image", margin: float = 0.05) -> "Image":
        """DELTA - difference between images

        img = img1.RMSDELTA(img2) calculates the difference images
        between a pair of images, ignoring edge pixels.
        
        By default, each of the four edges is 5% of the corresponding
        image dimension. The MARGIN parameter overrides the default.
        """
        Y, X = self.shape
        DY = int(margin*Y)
        DX = int(margin*X)
        return self[DY:-DY, DX:-DX] - img2[DY:-DY, DX:-DX]
