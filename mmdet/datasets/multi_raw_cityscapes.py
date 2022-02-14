# Copyright (c) OpenMMLab. All rights reserved.
from .builder import DATASETS
from .multi_raw import MultiRAWDataset


@DATASETS.register_module()
class MultiRAWCityscapesDataset(MultiRAWDataset):
    CLASSES = ('person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
               'bicycle')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
