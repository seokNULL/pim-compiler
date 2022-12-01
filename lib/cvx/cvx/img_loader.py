"""
Image loader module on top of OpenCV and numpy

Authors: Jorn Tuyls
"""

import cv2
import numpy as np
import logging
logger = logging.getLogger('cvx')

class ImgLoader(object):

    """
    Responsible for loading image data from file

    Arguments
    ---------
    layout: str
        the layout of the images to be returned (only NHWC supported)
    color_format: str
        the color format in which the image is loaded 
        (only RGB and BGR supported for now)
    """

    def __init__(self,
                 layout='NHWC',
                 color_format='RGB'):
        
        if layout not in ['NHWC']:
            raise NotImplementedError("Images can only be loaded in `NHWC`"\
                " format for now but got {}".format(layout))
        if color_format not in ['RGB', 'BGR']:
            raise ValueError("Unsupported color format: {}. The data"\
                " loader only handles `RGB` and `BGR` for now.")
        
        self.layout = layout
        self.color_format = color_format
    
    def load(self, img_paths):
        # type: (List[str]) -> numpy.ndarray
        """
        Load the image data from the specified file paths

        Returns
        -------
        imgs: numpy.ndarray
            the images read from the provided image paths in specified
            layout and color format.
        """

        imgs = []
        for p in img_paths:
            img = cv2.imread(p)

            if self.color_format == 'RGB':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            #if self.layout == 'NCHW':
            #    img = np.transpose(img, (2,0,1)) # HWC -> CHW

            imgs.append(img.astype(np.float32))
        
        # NHWC
        return imgs


