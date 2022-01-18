# Copyright (c) OpenMMLab. All rights reserved.
from .builder import DATASETS
from .multi_raw import MultiRAWDataset


@DATASETS.register_module()
class MultiRAWCarOnlyDataset(MultiRAWDataset):
    CLASSES = ('car',)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


