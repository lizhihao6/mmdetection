# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import xml.etree.ElementTree as ET
from collections import OrderedDict

import mmcv
import torch
from mmcv.utils import print_log

from mmdet.core import eval_map, eval_recalls
import numpy as np

from .builder import DATASETS
from .multi_raw import MultiRAWDataset
from .pipelines import Compose

from mmcomp.apis.inference import init_compressor
from mmcomp.utils import rearrange, inverse_rearrange

@DATASETS.register_module()
class MultiRAWCompDataset(MultiRAWDataset):
    CLASSES = ('car', 'person', 'traffic sign', 'traffic light')

    def __init__(self, **kwargs):
        self.load_pipeline = Compose(kwargs.pop('load_pipeline'))
        self.comp_model = init_compressor(kwargs.pop('comp_config'),
                                          kwargs.pop('comp_checkpoint'),
                                          device=0)
        super().__init__(**kwargs)
        
    def prepare_test_img(self, idx):
        """Get testing data  after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by \
                pipeline.
        """
        img_info = self.data_infos[idx]
        results = dict(img_info=img_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        results = self.load_pipeline(results)

        # convert img to compression
        img = results['img']
        if len(img.shape) == 2:
            img = rearrange(img).astype(np.float32)
            blc, saturate = results['black_level'], results['white_level']
            img = (img-blc)/(saturate-blc)
            img = np.clip(img, 0, 1)
        else:
            img = img[..., ::-1]
            img = img.astype(np.float32)/255.
        img = torch.from_numpy(img[None].copy()).permute([0, 3, 1, 2]).cuda()
        # compression
        with torch.no_grad():
            comp_results = self.comp_model(img=img, return_loss=False, return_image=True)
        img = comp_results['rec_img']
        # convert img to detection
        img = img[0].permute([1, 2, 0]).detach().cpu().numpy()
        img = np.clip(img, 0, 1)
        if img.shape[2] == 4:
            img = inverse_rearrange(img)
            blc, saturate = results['black_level'], results['white_level']
            img = img*(saturate-blc)+blc
        else:
            img = img[..., ::-1]
            img *= 255.
        results['img'] = np.round(img)
        resutls = self.pipeline(results)
        resutls['bpp'] = float(comp_results['bpp'])
        resutls['psnr'] = float(comp_results['psnr'])
        return resutls

    def prepare_train_img(self, idx):
        raise NotImplementedError
