"""
Module for testing ImgProcessor

Authors: Jorn Tuyls
"""

import unittest
import numpy as np

import sys
import logging

from cvx.img_processor import ImgProcessor

logger = logging.getLogger('cvx')
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.DEBUG)


class TestImgProcessor(unittest.TestCase):

    def test_scale_transpose(self):
        logger.debug("Test scale and tranpose")
        # NCHW
        imgs = np.transpose(
            np.reshape(np.array([
                [[10,10],
                [50,10]],
                [[30,50],
                [10,90]],
                [[20, 0],
                [0, 0]]
            ]), (1,3,2,2)),
            (0,2,3,1)
        ) # NCHW -> NHWC

        img_processor = ImgProcessor('scale-2.0__transpose-2,0,1')
        res = img_processor.execute(imgs)

        expected_outpt = np.transpose(2.0 * imgs, (0,3,1,2)) # NHWC -> NCHW
        
        np.testing.assert_array_equal(res, expected_outpt)

    def test_crop_transpose(self):
        logger.debug("Test crop and tranpose")
        # NCHW
        imgs = np.transpose(
            np.reshape(np.array([
                [[10,10,0],
                [50,10,0],
                [0,0,0]],
                [[30,50,0],
                [10,90,0],
                [0,0,0]],
                [[20,0,0],
                [0,0,0],
                [0,0,0]]
            ]), (1,3,3,3)),
            (0,2,3,1)
        )

        img_processor = ImgProcessor('crop-0,2-0,2-1,3__transpose-2,0,1')
        res = img_processor.execute(imgs)

        expected_outpt = np.reshape(np.array([
            [[30,50],
            [10,90]],
            [[20, 0],
            [0, 0]]
        ]), (1,2,2,2)) # NCHW
        
        np.testing.assert_array_equal(res, expected_outpt)
    



if __name__ == '__main__':
    unittest.main()