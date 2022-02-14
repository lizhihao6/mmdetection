# Copyright (c) OpenMMLab. All rights reserved.
import csv
import os.path as osp

import cv2
import numpy as np
import torch
from mmcomp.apis.inference import init_compressor
from mmcomp.utils import rearrange, inverse_rearrange

from .builder import DATASETS
from .multi_raw import MultiRAWDataset
from .pipelines import Compose


@DATASETS.register_module()
class MultiRAWCompDataset(MultiRAWDataset):
    CLASSES = ('car', 'person', 'traffic sign', 'traffic light')

    def __init__(self, **kwargs):
        self.load_pipeline = Compose(kwargs.pop('load_pipeline'))
        self.comp_model = init_compressor(kwargs.pop('comp_config'),
                                          kwargs.pop('comp_checkpoint'),
                                          device=0)
        self.prefix = kwargs.pop('prefix')
        if self.prefix is not None:
            bpp_psnr_dict = {}
            with open(osp.join(self.prefix, 'comp_results.csv')) as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader[1:]:
                    bpp_psnr_dict[row['filename']] = [float(row['bpp']), float(row['psnr'])]
            for d in self.data_infos:
                filename = osp.basname(d['filename'])
                d['filename'] = osp.join(self.prefix, f'{filename.split(".")[0]}.png')
                d['bpp'] = bpp_psnr_dict[filename][0]
                d['psnr'] = bpp_psnr_dict[filename][1]
        super().__init__(**kwargs)

    def img_format(self, img, results):
        if len(img.shape) == 2:
            img = rearrange(img).astype(np.float32)
            blc, saturate = results['black_level'], results['white_level']
            img = (img - blc) / (saturate - blc)
            img = np.clip(img, 0, 1)
        else:
            img = img[..., ::-1]
            img = img.astype(np.float32) / 255.
        return img

    def img_deformat(self, img, results):
        img = np.clip(img, 0, 1)
        if img.shape[2] == 4:
            img = inverse_rearrange(img)
            blc, saturate = results['black_level'], results['white_level']
            img = img * (saturate - blc) + blc
        else:
            img = img[..., ::-1] * 255
        return np.round(img)

    def compression(self, results):
        # convert img to compression
        img = results['img']
        img = self.img_format(img, results)
        img_metas = [dict(black_level=results['black_level'],
                          white_level=results['white_level'])] if 'black_level' in results else [{}]
        img = torch.from_numpy(img[None].copy()).permute([0, 3, 1, 2]).cuda()
        # compression
        with torch.no_grad():
            comp_results = self.comp_model(img=img, return_loss=False, return_image=True, img_metas=img_metas)
        img = comp_results['rec_img']
        # convert img to detection
        if isinstance(img, torch.Tensor):
            img = img[0].permute([1, 2, 0]).detach().cpu().numpy()
        else:
            img = cv2.imread(img, cv2.IMREAD_UNCHANGED).astype(np.float32)
            img = self.img_format(img, results)

        results['img'] = self.img_deformat(img, results)
        results = self.pipeline(results)
        results['bpp'] = float(comp_results['bpp'])
        results['psnr'] = float(comp_results['psnr'])
        return results

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
        if self.prefix is None:
            results = self.load_pipeline(results)
            results = self.compression(results)
        else:
            results = self.load_pipeline(results)
            results['bpp'] = img_info['bpp']
            results['psnr'] = img_info['psnr']
        return results

    def prepare_train_img(self, idx):
        raise NotImplementedError
