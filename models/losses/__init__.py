# -*- coding: utf-8 -*-
# @Time    : 2021/9/29 10:02
# @Author  : vivian
import copy
from .DB_loss import DBLoss, DualDBLoss

__all__ = ['build_loss']
support_loss = ['DBLoss', 'DualDBLoss']


def build_loss(config):
    copy_config = copy.deepcopy(config)
    loss_type = copy_config.pop('type')
    assert loss_type in support_loss, f'all support loss is {support_loss}'
    criterion = eval(loss_type)(**copy_config)
    return criterion
