from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable

import torch
import torch.nn as nn

from model.resnet import resnet18,resnet18_1


def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x

class Encoder(nn.Module):
    def __init__(self, task: str = ""):
        super().__init__()
        self.task = task
        custom_pretrained_weights = "/home/xiaridehehe/ownProgram/PDSENet/weight/ResNet18-model-new.pth"
        self.resnet18 = resnet18_1(pretrained=False, custom_pretrained_weights=custom_pretrained_weights)  # 使用默认预训练权重
        self.resnet = resnet18_1(pretrained=False, custom_pretrained_weights=None)
        self.drop = nn.Dropout(p=0.01)

    def capture(self, x):
        x = self.resnet18.pre(x)

        x2 = self.drop(self.resnet18.layer1(x))
        x3 = self.resnet18.layer2(x2)
        x4 = self.resnet18.layer3(x3)
        x5 = self.resnet18.layer4(x4)

        return x5

    def detail_capture(self, x):
        x = self.resnet.pre(x)

        x2 = self.drop(self.resnet.layer1(x))
        x3 = self.resnet.layer2(x2)
        x4 = self.resnet.layer3(x3)

        return [x2, x3, x4]

    def forward(self, x, y):
        # print("x", x.shape)
        v_x = self.capture(x)
        v_y = self.capture(y)
        # print("x4", v_x.shape, v_y.shape)

        c_x = self.detail_capture(x)
        c_y = self.detail_capture(y)

        return c_x + [v_x], c_y + [v_y]
