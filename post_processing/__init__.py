# -*- coding: utf-8 -*-
# @Time    : 2021/9/29 14:09
# @Author  : vivian

from .seg_detector_representer import SegDetectorRepresenter


def get_post_processing(config):
    try:
        cls = eval(config['type'])(**config['args'])
        return cls
    except:
        return None