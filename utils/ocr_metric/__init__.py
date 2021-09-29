# -*- coding: utf-8 -*-
# @Time    : 2021/9/29 14:09
# @Author  : vivian
from .icdar2015 import QuadMetric


def get_metric(config):
    try:
        if 'args' not in config:
            args = {}
        else:
            args = config['args']
        if isinstance(args, dict):
            cls = eval(config['type'])(**args)
        else:
            cls = eval(config['type'])(args)
        return cls
    except:
        return None