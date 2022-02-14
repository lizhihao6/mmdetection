# Copyright (c) OpenMMLab. All rights reserved.

from collections import OrderedDict

import torch
from mmgen.models import build_module

from .mobilenet_v2_raw import MobileNetV2RAW
from ..builder import BACKBONES


@BACKBONES.register_module()
class MobileNetV2RAWganISP(MobileNetV2RAW):

    def __init__(self, **kwargs):
        self.ganISP_pretrained = kwargs.pop('ganISP_pretrained')
        super().__init__(**kwargs)
        cfg = dict(type='inverseISP')
        self.ganISP = build_module(cfg)
        self.loaded = False

    def forward(self, x):
        """Forward function."""
        if not self.loaded:
            self.load_ganISP_checkpoint(self.ganISP_pretrained)
            self.loaded = True

        if self.training:
            with torch.no_grad():
                x = self.ganISP(x)
        return super().forward(x)

    def load_ganISP_checkpoint(self, checkpoint_path):
        """Load ganISP checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            if k.startswith('generator.'):
                state_dict[k[len('generator.'):]] = v
        self.ganISP.load_state_dict(state_dict, strict=True)
        self.ganISP.eval()
