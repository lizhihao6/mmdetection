# Copyright (c) OpenMMLab. All rights reserved.
from .detr import DETR
from ..builder import DETECTORS


@DETECTORS.register_module()
class DeformableDETR(DETR):

    def __init__(self, *args, **kwargs):
        super(DETR, self).__init__(*args, **kwargs)
