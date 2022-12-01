"""
CV postprocessing operations

Authors: Jorn Tuyls
"""

import cv2
import numpy as np

from .tools import softmax

def central_crop(height, width, channels):
    # type: (str/int, str/int, str/int) -> Function
    """
    Return a wrapper function that takes in an image and centrally crops 
    an image of provided height, width and channels
    """
    height, width, channels = int(height), int(width), int(channels)

    def _central_crop(img):
        # !! img should be in HWC layout
        img_h, img_w, img_c = img.shape

        if height > img_h:
            raise ValueError("Provided crop height is larger than provided"\
                " image height.")
        if width > img_w:
            raise ValueError("Provided crop width is larger than provided"\
                " image width.")
        if channels > img_c:
            raise ValueError("Provided crop channels value is larger than"
                " provided image channels.")
        
        start_h = int((img_h - height) / 2)
        end_h = start_h + height
        start_w = int((img_w - width) / 2)
        end_w = start_w + width
        start_c = int((img_c - channels) / 2)
        end_c = start_c + channels

        return img[start_h:end_h, start_w:end_w, start_c:end_c]

    return _central_crop