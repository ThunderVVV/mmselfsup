# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.runner import BaseModule

from ..builder import NECKS


@NECKS.register_module()
class NoneNeck(BaseModule):
    """The linear neck: fc only.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        with_avg_pool (bool): Whether to apply the global
            average pooling after backbone. Defaults to True.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 with_avg_pool=True,
                 init_cfg=None):
        super(NoneNeck, self).__init__(init_cfg)
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        assert len(x) == 1
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        return [x.view(x.size(0), -1)]
