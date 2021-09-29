# -*- coding: utf-8 -*-
# @Time    : 2021/9/29 14:09
# @Author  : vivian
from torch import nn

from models.losses.basic_loss import BalanceCrossEntropyLoss, MaskL1Loss, DiceLoss


class DBLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=10, ohem_ratio=3, reduction='mean', eps=1e-6):
        """
        Implement PSE Loss.
        :param alpha: binary_map loss 前面的系数
        :param beta: threshold_map loss 前面的系数
        :param ohem_ratio: OHEM的比例
        :param reduction: 'mean' or 'sum'对 batch里的loss 算均值或求和
        """
        super().__init__()
        assert reduction in ['mean', 'sum'], " reduction must in ['mean','sum']"
        self.alpha = alpha
        self.beta = beta
        self.bce_loss = BalanceCrossEntropyLoss(negative_ratio=ohem_ratio)
        self.dice_loss = DiceLoss(eps=eps)
        self.l1_loss = MaskL1Loss(eps=eps)
        self.ohem_ratio = ohem_ratio
        self.reduction = reduction

    def forward(self, pred, batch):
        shrink_maps = pred[:, 0, :, :]
        threshold_maps = pred[:, 1, :, :]
        binary_maps = pred[:, 2, :, :]

        loss_shrink_maps = self.bce_loss(shrink_maps, batch['shrink_map'], batch['shrink_mask'])
        loss_threshold_maps = self.l1_loss(threshold_maps, batch['threshold_map'], batch['threshold_mask'])
        metrics = dict(loss_shrink_maps=loss_shrink_maps, loss_threshold_maps=loss_threshold_maps)
        if pred.size()[1] > 2:
            loss_binary_maps = self.dice_loss(binary_maps, batch['shrink_map'], batch['shrink_mask'])
            metrics['loss_binary_maps'] = loss_binary_maps
            loss_all = self.alpha * loss_shrink_maps + self.beta * loss_threshold_maps + loss_binary_maps
            metrics['loss'] = loss_all
        else:
            metrics['loss'] = loss_shrink_maps
        return metrics


class DualDBLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=10, ohem_ratio=3, reduction='mean', eps=1e-6):
        """
        Implement Dual DB Loss.
        :param alpha: binary_map loss 前面的系数
        :param beta: threshold_map loss 前面的系数
        :param ohem_ratio: OHEM的比例
        :param reduction: 'mean' or 'sum'对 batch里的loss 算均值或求和
        """
        super().__init__()
        assert reduction in ['mean', 'sum'], " reduction must in ['mean','sum']"
        self.alpha = alpha
        self.beta = beta
        self.fore_bce_loss = BalanceCrossEntropyLoss(negative_ratio=ohem_ratio)
        self.fore_dice_loss = DiceLoss(eps=eps)
        self.fore_l1_loss = MaskL1Loss(eps=eps)

        self.back_bce_loss = BalanceCrossEntropyLoss(negative_ratio=ohem_ratio)
        self.back_dice_loss = DiceLoss(eps=eps)
        self.back_l1_loss = MaskL1Loss(eps=eps)

        self.ohem_ratio = ohem_ratio
        self.reduction = reduction



    def forward(self, pred, batch):
        fore_shrink_maps = pred[:, 0, :, :]
        fore_threshold_maps = pred[:, 1, :, :]
        fore_binary_maps = pred[:, 2, :, :]

        back_shrink_maps = pred[:, 3, :, :]
        back_threshold_maps = pred[:, 4, :, :]
        back_binary_maps = pred[:, 5, :, :]


        fore_loss_shrink_maps = self.fore_bce_loss(fore_shrink_maps, batch['fore_shrink_map'], batch['fore_shrink_mask'])
        fore_loss_threshold_maps = self.fore_l1_loss(fore_threshold_maps, batch['fore_threshold_map'], batch['fore_threshold_mask'])

        back_loss_shrink_maps = self.back_bce_loss(back_shrink_maps, batch['back_shrink_map'],
                                                   batch['back_shrink_mask'])
        back_loss_threshold_maps = self.back_l1_loss(back_threshold_maps, batch['back_threshold_map'],
                                                batch['back_threshold_mask'])

        metrics = dict(fore_loss_shrink_maps=fore_loss_shrink_maps, fore_loss_threshold_maps=fore_loss_threshold_maps,
                       back_loss_shrink_maps=back_loss_shrink_maps, back_loss_threshold_maps=back_loss_threshold_maps)

        if pred.size()[1] > 4:
            fore_loss_binary_maps = self.fore_dice_loss(fore_binary_maps, batch['fore_shrink_map'], batch['fore_shrink_mask'])
            back_loss_binary_maps = self.back_dice_loss(back_binary_maps, batch['back_shrink_map'],
                                                        batch['back_shrink_mask'])

            metrics['fore_loss_binary_maps'] = fore_loss_binary_maps
            metrics['back_loss_binary_maps'] = back_loss_binary_maps
            fore_loss = self.alpha * fore_loss_shrink_maps + self.beta * fore_loss_threshold_maps + fore_loss_binary_maps
            back_loss = self.alpha * back_loss_shrink_maps + self.beta * back_loss_threshold_maps + back_loss_binary_maps
            loss_all = (fore_loss + back_loss) / 2

            metrics['loss'] = loss_all
        else:
            metrics['loss'] = (fore_loss_shrink_maps + back_loss_shrink_maps) /2
        return metrics