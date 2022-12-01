
"""
CV base operations

Authors: Jorn Tuyls
"""

import cv2
import numpy as np


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
            raise ValueError("Provided crop height is larger than provided"
                             " image height.")
        if width > img_w:
            raise ValueError("Provided crop width is larger than provided"
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


def swap_channels(axes):
    # type: (List[str/[int]) -> Function
    """
    Return a wrapper function takes in an image and swaps channels
    according to the provided axes argument.

    E.g RGB -> BGR or BGR -> RGB
    """

    def _swap_channels(img):
        # Img is in HWC
        channels = [img[:, :, i] for i in [0, 1, 2]]

        return np.stack(tuple([channels[a] for a in axes]), axis=2)

    return _transpose


def crop(height, width, channels):
    # type: (List[str/int], List[str/int], List[str,int]) -> Function
    """
    Return a wrapper function that takes in an image and crops it according to
    provided height, width and channel boundaries.
    """

    assert(len(height) == 2 and len(width) == 2 and len(channels) == 2)

    start_h, end_h = int(height[0]), int(height[1])
    start_w, end_w = int(width[0]), int(width[1])
    start_c, end_c = int(channels[0]), int(channels[1])

    def _crop(img):
        # img should be in HWC layout
        return img[start_h:end_h, start_w:end_w, start_c:end_c]

    return _crop


def flip(axes):
    # type: (List[str/int]) -> Function
    """
    Return a wrapper function that takes in an image and flips the
    provided axes
    """

    axes = tuple([int(a) for a in axes])

    def _flip(img):
        return np.flip(img, axes)

    return _flip


def normalize(means, stdevs):
    # type: (List[str/int/float], List[str/int/float]) -> Function
    """
    Return a wrapper function to normalize an image according to provided
    means and standard deviations.
    """
    assert(len(means) == len(stdevs))

    means = [float(mean) for mean in means]
    stdevs = [float(stdev) for stdev in stdevs]

    def _normalize(img):
        # img should be in HWC layout
        assert(img.shape[2] == len(means))
        return (img - means) / stdevs

    return _normalize


def resize(size, interpolation='INTER_LINEAR',
           keep_aspect_ratio=False, pad_values=None):
    # type: (List[str/int], str, bool, List[str/int]) -> Function
    """
    Return a wrapper function to resize an image to provided size.

    Arguments
    ---------
    size: List[str/int]
        the new size as a width, height tuple
    interpolation: str
        the interpolation algorithm to be used
    keep_aspect_ratio: bool
        whether to keep aspect ratio and add padding
    pad_values:
        the padding values to be used
    """
    assert len(size) == 2

    size = [int(dim) if dim not in ['?', 'None', None] else None
            for dim in size]
    assert size != [None, None]

    keep_aspect_ratio = (keep_aspect_ratio in ["1", 1, "true", "True", True])
    assert not keep_aspect_ratio or (None not in size)
    pad_values = [float(p) for p in pad_values] \
        if pad_values is not None else None

    interpolations = {
        'INTER_NEAREST': cv2.INTER_NEAREST,
        'INTER_LINEAR': cv2.INTER_LINEAR,
        'INTER_AREA': cv2.INTER_AREA,
        'INTER_CUBIC': cv2.INTER_CUBIC,
        'INTER_LANCZOS4': cv2.INTER_LANCZOS4
    }

    def get_aspect_ratio_size(height, width, new_height, new_width):
        scale = min(new_width / float(width), new_height / float(height))
        return (int(height * scale), int(width * scale))

    def _resize(img):
        # !! img should be in HWC format
        if img.dtype not in ['float32']:
            raise ValueError("OpenCV resize operator expects imput array"
                             " to have float32 data type but got: {}"
                             .format(img.dtype))

        if size[0] is None:
            height_aspect_ratio = size[1] / float(img.shape[0])
            size[0] = int(img.shape[1] * height_aspect_ratio)
            res = cv2.resize(img, tuple(size),
                             interpolation=interpolations[interpolation])
        elif size[1] is None:
            width_aspect_ratio = size[0] / float(img.shape[1])
            size[1] = int(img.shape[0] * width_aspect_ratio)
            res = cv2.resize(img, tuple(size),
                             interpolation=interpolations[interpolation])
        elif keep_aspect_ratio:
            height, width = img.shape[0], img.shape[1]

            new_height, new_width = size[1], size[0]

            resize_height, resize_width = \
                get_aspect_ratio_size(height, width, new_height, new_width)

            res = cv2.resize(img, (resize_width, resize_height),
                             interpolation=interpolations[interpolation])

            delta_h = new_height - resize_height
            delta_w = new_width - resize_width
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)

            res = cv2.copyMakeBorder(res, top, bottom, left, right,
                                     cv2.BORDER_CONSTANT,
                                     value=pad_values)
        else:
            res = cv2.resize(img, tuple(size),
                             interpolation=interpolations[interpolation])

        return res

    return _resize


def resize_smallest_side(size, interpolation='INTER_LINEAR'):
    # type: (str/int) -> Function
    """
    Return a wrapper function to resize an image so that the smallest size is
    equal to the provided size.

    Arguments
    ---------
    size: str/int
        the new size of the smallest side of the image
    """
    size = int(size)

    interpolations = {
        'INTER_NEAREST': cv2.INTER_NEAREST,
        'INTER_LINEAR': cv2.INTER_LINEAR,
        'INTER_AREA': cv2.INTER_AREA,
        'INTER_CUBIC': cv2.INTER_CUBIC,
        'INTER_LANCZOS4': cv2.INTER_LANCZOS4
    }

    def get_size(height, width, aspect_ratio):
        return (int(width * aspect_ratio), int(height * aspect_ratio))

    def _resize_smallest_side(img):
        # !! img should be in HWC format
        if img.dtype not in ['float32']:
            raise ValueError("OpenCV resize operator expects imput array"
                             " to have float32 data type but got: {}"
                             .format(img.dtype))

        smallest_side_size = img.shape[0] if img.shape[0] < img.shape[1] \
            else img.shape[1]
        aspect_ratio = size / float(smallest_side_size)

        new_size = get_size(img.shape[0], img.shape[1], aspect_ratio)

        return cv2.resize(img, new_size,
                          interpolation=interpolations[interpolation])

    return _resize_smallest_side


def resize_largest_side(size, interpolation='INTER_LINEAR'):
    # type: (str/int, str) -> Function
    """
    Return a wrapper function to resize an image so that the sizes are
    a multiple of the provided value

    Arguments
    ---------
    size: str/int
        the size of the largest size to be used for resizing
    interpolation: str
        which interpolation algorithm to use
    """
    size = int(size)

    interpolations = {
        'INTER_NEAREST': cv2.INTER_NEAREST,
        'INTER_LINEAR': cv2.INTER_LINEAR,
        'INTER_AREA': cv2.INTER_AREA,
        'INTER_CUBIC': cv2.INTER_CUBIC,
        'INTER_LANCZOS4': cv2.INTER_LANCZOS4
    }

    def get_size(height, width, aspect_ratio):
        return (int(width * aspect_ratio), int(height * aspect_ratio))

    def _resize_largest_side(img):
        # !! img should be in HWC format
        if img.dtype not in ['float32']:
            raise ValueError("OpenCV resize operator expects imput array"
                             " to have float32 data type but got: {}"
                             .format(img.dtype))

        largest_side_size = img.shape[0] if img.shape[0] > img.shape[1] \
            else img.shape[1]
        aspect_ratio = size / float(largest_side_size)

        new_size = get_size(img.shape[0], img.shape[1], aspect_ratio)

        return cv2.resize(img, new_size,
                          interpolation=interpolations[interpolation])

    return _resize_largest_side


def resize_to_multiple(multiple, interpolation='INTER_LINEAR',
                       keep_aspect_ratio=True,
                       pad_values=None):
    # type: (str/int, str, bool, List[str/int]) -> Function
    """
    Return a wrapper function to resize an image so that the sizes are
    a multiple of the provided value

    Arguments
    ---------
    multiple: str/int
        the multiple to be used for resizing
    interpolation: str
        which interpolation algorithm to use
    keep_aspect_ratio: bool
        whether to keep the original aspect ratio
    pad_values: List[str/int]
        the values to be used for padding
    """
    multiple = int(multiple)
    keep_aspect_ratio = (keep_aspect_ratio in ["1", 1, "true", "True", True])
    pad_values = [float(p) for p in pad_values] \
        if pad_values is not None else None

    interpolations = {
        'INTER_NEAREST': cv2.INTER_NEAREST,
        'INTER_LINEAR': cv2.INTER_LINEAR,
        'INTER_AREA': cv2.INTER_AREA,
        'INTER_CUBIC': cv2.INTER_CUBIC,
        'INTER_LANCZOS4': cv2.INTER_LANCZOS4
    }

    def get_size(height, width, new_height, new_width):
        scale = min(new_width / float(width), new_height / float(height))
        return (int(height * scale), int(width * scale))

    def _resize_to_multiple(img):
        # !! img should be in HWC format
        if img.dtype not in ['float32']:
            raise ValueError("OpenCV resize operator expects imput array"
                             " to have float32 data type but got: {}"
                             .format(img.dtype))
        height, width = img.shape[0], img.shape[1]

        new_height = height - (height % multiple)
        new_width = width - (width % multiple)

        if keep_aspect_ratio:
            resize_height, resize_width = \
                get_size(height, width, new_height, new_width)

            res = cv2.resize(img, (resize_width, resize_height),
                             interpolation=interpolations[interpolation])

            delta_h = new_height - resize_height
            delta_w = new_width - resize_width
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)

            res = cv2.copyMakeBorder(res, top, bottom, left, right,
                                     cv2.BORDER_CONSTANT,
                                     value=pad_values)

        else:
            res = cv2.resize(img, (new_height, new_width),
                             interpolation=interpolations[interpolation])

        return res

    return _resize_to_multiple


def scale(scale):
    # type: (str/int/float) -> Function
    """
    Return a wrapper function that takes in an image and scales it according to
    the provided scale argument.
    """

    def _scale(img):
        return img * float(scale)

    return _scale


def subtract(values):
    # type: (List[str/int/float]) -> Function
    """
    Return a wrapper function that takes in an image and subtracts the provided
    values
    """
    values = [float(v) for v in values]

    def _subtract(img):
        return img - values

    return _subtract


def transpose(axes):
    # type: (List[str/[int]) -> Function
    """
    Return a wrapper function takes in an image and transposes it according to
    the provided axes argument.
    """

    axes = [int(axis) for axis in axes]

    def _transpose(img):
        return np.transpose(img, axes=axes)

    return _transpose
