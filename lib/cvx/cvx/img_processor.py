"""
Image operation module on top of OpenCV and numpy

Authors: Jorn Tuyls
"""

import cv2
import numpy as np
import logging

from . import op as ops

logger = logging.getLogger('cvx')


class ImgProcessor(object):

    """
    Responsible for processing image data

    Arguments
    ---------
    proc_key: str
        the processing key indicating what data processing to perform
    """

    def __init__(self, proc_key, layout='NHWC'):

        if layout not in ['NHWC']:
            raise ValueError("Only `NHWC` layout is supported for now"
                             " for ImgProcessor but got: {}"
                             .format(layout))

        self.proc_key = proc_key
        self.layout = layout

        self.proc_ops = []
        self.proc_funcs = {}
        self._parse_processing_funcs(proc_key)

    def _parse_processing_funcs(self, proc_key):
        # type: (str) -> None
        """
        Parse the processing key and translate into image transformation
        functions

        Supported structure of the processing key is:
        prepfuncname1-arg1-arg2val1,...,arg2valn-...-argn
            __prepfuncname2-arg1-...-argn_...
        """

        proc_ops = {
            'central_crop': ops.central_crop,
            'chswap': ops.swap_channels,
            'crop': ops.crop,
            'flip': ops.flip,
            'normalize': ops.normalize,
            'resize': ops.resize,
            'resize_smallest_side': ops.resize_smallest_side,
            'resize_to_multiple': ops.resize_to_multiple,
            'scale': ops.scale,
            'subtract': ops.subtract,
            'transpose': ops.transpose
        }

        proc_funcs = proc_key.split("__")
        for idx, proc_func in enumerate(proc_funcs):
            proc_func_parsed = proc_func.split("-")

            proc_func_name = proc_func_parsed[0]
            if proc_func_name not in proc_ops:
                raise NotImplementedError("Unknown processing function: {}"
                                          .format(proc_func_name))

            proc_func_args = \
                [(pfa.split(',') if ',' in pfa else pfa)
                 for pfa in proc_func_parsed[1:]]
            logger.debug(proc_func_args)

            self.proc_ops.append(idx)
            self.proc_funcs[idx] = proc_ops[proc_func_name](*proc_func_args)

    def add_processing_func(self, func):
        # type: (Function) -> None
        """
        Add custom preprocessing function
        """
        raise NotImplementedError("")

    def execute(self, X):
        # type: (numpy.ndarray/List[numpy.ndarray]) -> numpy.ndarray
        """
        Process the specified image data

        Arguments
        ---------
        X: numpy.ndarray
            the image data in NCHW or NHWC structure
        """
        logger.debug(self.proc_ops)

        if isinstance(X, np.ndarray) and X.ndim != 4:
            raise ValueError("Image processor expects image data as"
                             " a numpy array with 4 dimensions (NHWC"
                             " or NCHW layout) but got array with {}"
                             " dimensions.".format(X.ndim))

        N = X.shape[0] if isinstance(X, np.ndarray) else len(X)

        res = []
        # Iterate over N dimension (batch)
        for i in range(N):
            img = X[i]

            for proc_func_name in self.proc_ops:
                logger.debug(proc_func_name)
                logger.debug("Shape before: {}".format(img.shape))

                img = self.proc_funcs[proc_func_name](img)

                logger.debug("Shape after: {}".format(img.shape))

            # HWC -> CHW if requested
            # if self.layout == 'CHW':
            #    img = np.transpose(img, (2,0,1)) # HWC -> CHW

            res.append(img)

        return np.array(res).astype(np.float32)
