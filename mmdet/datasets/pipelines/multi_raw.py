# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp

import numpy as np
import rawpy
from imageio import imread
from skimage.exposure import equalize_hist

from ..builder import PIPELINES

try:
    from panopticapi.utils import rgb2id
except ImportError:
    rgb2id = None


@PIPELINES.register_module()
class LoadRAWFromFile:
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).
    """

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        if filename.endswith('TIF'):
            img = imread(filename)
        else:
            with rawpy.imread(filename) as f:
                img = f.raw_image_visible.copy()

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img.astype(np.float32)
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']

        with open(results['img_info']['meta_path'], 'r') as f:
            results = {**results, **json.load(f)}
        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}()'
        return repr_str


@PIPELINES.register_module()
class Rearrange:
    """Rearrange RAW to four channels.
    """

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        img = results['img']
        h, w = img.shape
        rearrange = np.zeros([h // 2, w // 2, 4], dtype=img.dtype)
        if results['bayer_pattern'] == 'rggb':
            img = img
        elif results['bayer_pattern'] == 'byyr':
            img = np.pad(img, (1, 0), (1, 0), mode='reflect')[:-1, :-1]
        else:
            raise NotImplementedError
        rearrange[..., 0] = img[0::2, 0::2]
        rearrange[..., 1] = img[0::2, 1::2]
        rearrange[..., 2] = img[1::2, 0::2]
        rearrange[..., 3] = img[1::2, 1::2]

        results['img'] = rearrange
        results['img_shape'] = rearrange.shape
        results['ori_shape'] = rearrange.shape
        """Resize bounding boxes with 0.5."""
        for key in results.get('bbox_fields', []):
            bboxes = results[key] * 0.5
            img_shape = results['img_shape']
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            results[key] = bboxes
        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}()'
        return repr_str


@PIPELINES.register_module()
class CamTosRGB:
    """Mapping Camera Color Space to sRGB space, add behind Rearrange
    """

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        img = results['img']
        assert img.shape[2] == 4
        # linear
        img = (img - results['black_level']) / (results['white_level'] - results['black_level'])
        # wb
        img[..., 0] *= results['white_balance'][0] / results['white_balance'][1]
        img[..., 3] *= results['white_balance'][2] / results['white_balance'][1]
        # cam -> srgb
        ccm = np.array(results['color_matrix'], dtype=np.float32).T
        r, g_r, g_b, b = np.split(img, 4, axis=2)
        _r = r * ccm[0, 0] + g_r * ccm[1, 0] * 0.5 + g_b * ccm[1, 0] * 0.5 + b * ccm[2, 0]
        _g_r = r * ccm[0, 1] + g_r * ccm[1, 1] + b * ccm[2, 1]
        _g_b = r * ccm[0, 1] + g_b * ccm[1, 1] + b * ccm[2, 1]
        _b = r * ccm[0, 2] + g_r * ccm[1, 2] * 0.5 + g_b * ccm[1, 2] * 0.5 + b * ccm[2, 2]
        # resale
        # white_level = results['white_level'] - results['black_level']
        # img = np.concatenate([_r, _g_r, _g_b, _b], axis=2)
        # img = np.clip(img, 0, 1)*white_level
        # results['img'] = img
        # results['black_level'] = 0
        # results['white_level'] = white_level
        results['img'] = np.clip(img, 0, 1)
        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}()'
        return repr_str


@PIPELINES.register_module()
class HDRSplit:
    """ Split HDR into two sub image
    """

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        img = results['img']
        h, w, c = img.shape
        rearrange = np.zeros([h, w, 8], dtype=img.dtype)
        blc, saturate = results['black_level'], results['white_level']
        rearrange[..., :4] = np.clip(img, blc, blc + 255) - blc
        rearrange[..., 4:] = (np.clip(img, blc + 255, None) + 1 - blc) / 256 - 1
        max_v = (saturate + 1 - blc) / 256 - 1
        max_v = np.array([255., 255., 255., 255., max_v, max_v, max_v, max_v], dtype=img.dtype).reshape([1, 1, 8])
        rearrange /= max_v
        results['img'] = rearrange
        results['img_shape'] = rearrange.shape
        results['ori_shape'] = rearrange.shape
        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}()'
        return repr_str


@PIPELINES.register_module()
class EqualizeHist:
    """Histogram equalization
    """

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        results['img'] = equalize_hist(results['img'], nbins=results['white_level'])
        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}()'
        return repr_str


@PIPELINES.register_module()
class RYYBtoRGGB:
    """Histogram equalization
    """

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        img = results['img']
        assert img.shape[2] == 4
        blc, wl = results['black_level'], results['white_level']
        r = (img[..., 0] - blc) / (wl - blc)
        yr = (img[..., 1] - blc) / (wl - blc)
        yb = (img[..., 2] - blc) / (wl - blc)
        r, yr, yb = np.clip(r, 0, 1), np.clip(yr, 0, 1), np.clip(yb, 0, 1)
        inflection = 0.9
        mask_yr = (np.maximum(yr-inflection, 0) / (1-inflection))**2
        mask_yb = (np.maximum(yb-inflection, 0) / (1-inflection))**2
        yr = yr - (1-mask_yr) * r
        yb = yb - (1-mask_yb) * r
        img[..., 1] = np.clip(yr, 0, 1)
        img[..., 2] = np.clip(yb, 0, 1)
        # gr = np.where((yr > r) & ((r + yr) < 0.99 * 2), yr - r, yr)
        # gb = np.where((yb > r) & ((r+yb) < 0.99*2), yb-r, yb)
        # img[..., 1] = gr * (wl - blc) + blc
        # img[..., 2] = gb*(wl-blc)+blc
        results['img'] = img
        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}()'
        return repr_str


@PIPELINES.register_module()
class RGGBtoRYYB:
    """Histogram equalization
    """

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        img = results['img']
        assert img.shape[2] == 4
        blc, wl = results['black_level'], results['white_level']
        r = (img[..., 0] - blc) / (wl - blc)
        gr = (img[..., 1] - blc) / (wl - blc)
        gb = (img[..., 2] - blc) / (wl - blc)
        yr, yb = np.clip(gr + r, 0, 1), np.clip(gb + r, 0, 1)
        img[..., 1] = yr * (wl - blc) + blc
        img[..., 2] = yb * (wl - blc) + blc
        results['img'] = img
        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}()'
        return repr_str
