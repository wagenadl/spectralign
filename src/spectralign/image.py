from . import funcs
import numpy as np
import cv2

from typing import Optional, List
from numpy.typing import ArrayLike


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

    
    def ascii(self, width: int = 40) -> List[str]:
        """ASCII - ASCII-art copy of an image
        img.ASCII() returns a copy of the image converted to ascii art at low
        resolution. Optional parameter WIDTH specifies the desired width of
        the output. Intended to get a quick overview for use in terminals.
        """
        chrs = """ ·-+⊞▤▦▩■"""
        M = len(chrs)
        Y, X = self.shape
        scl = int(np.round(X/width+.99))
        w1 = X // scl
        h1 = Y // scl
        img = (self[:h1*scl,:w1*scl].reshape(h1, scl, w1, scl).mean(-1).mean(-2).view(np.ndarray)*M).astype(int)
        img[img<0] = 0
        img[img>=M] = M - 1
        lines = []
        for row in img:
            lines.append("".join([chrs[pix] for pix in row]))
        return lines

    def __repr__(self):
        if len(self.shape)==2:
            return f"""Image:
  {"\n  ".join(self.ascii())}
Size: {self.shape[1]} x {self.shape[0]} pixels
Min:  {self.min():.3f}
Max:  {self.max():.3f}
Mean: {self.mean():.3f}
SD:   {self.std():.3f}"""
        elif len(self.shape)==0:
            return self.view(np.ndarray).flatten()[0].__repr__()
        else:
            return self.view(np.ndarray).__repr__()
    
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
