#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn
import math

class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)
        
        # 中心点的距离平方
        pred_center = pred[:, :2]
        target_center = target[:, :2]
        center_distance = torch.sum((pred_center - target_center) ** 2, dim=1) # rho2

        # 计算包含两个框所有点的最小闭包区域的对角线长度的平方 c2
        c_tl = torch.min(pred[:, :2] - pred[:, 2:] / 2, target[:, :2] - target[:, 2:] / 2)
        c_br = torch.max(pred[:, :2] + pred[:, 2:] / 2, target[:, :2] + target[:, 2:] / 2)
        c2 = torch.sum((c_br - c_tl) ** 2, dim=1)

        # 计算宽高比的一致性
        v = (4 / (math.pi ** 2)) * torch.pow(
            torch.atan(target[:, 3] / target[:, 2]) - torch.atan(pred[:, 3] / pred[:, 2]), 2
        )

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "ciou":
            # 计算CIoU损失
            # alpha是权衡系数，用于调整v的影响
            with torch.no_grad():
                alpha = v / ((1 - iou) + v + 1e-16)
            ciou = iou - (center_distance / (c2 + 1e-16)) - alpha * v
            loss = 1 - ciou
        elif self.loss_type == "ciou_log":
            # print("Now we use coiu_log!!!")
            lambda_val = 1.0
            # 计算交集的宽度和高度
            inter_width = torch.clamp(br[:, 0] - tl[:, 0], min=0)
            inter_height = torch.clamp(br[:, 1] - tl[:, 1], min=0)

            # 计算交集的对数面积
            log_area_intersection = torch.log((inter_width + lambda_val) * (inter_height + lambda_val) + 1e-16)

            # 计算预测框的对数面积
            log_area_pred = torch.log((pred[:, 2] + lambda_val) * (pred[:, 3] + lambda_val) + 1e-16)

            # 计算真实框的对数面积
            log_area_target = torch.log((target[:, 2] + lambda_val) * (target[:, 3] + lambda_val) + 1e-16)

            # 计算并集的对数面积，这里使用log-sum-exp技巧进行数值稳定的并集面积计算
            log_area_union = torch.logsumexp(torch.stack((log_area_pred, log_area_target), dim=0),
                                             dim=0) - log_area_intersection

            # 最终的对数IOU
            iou_log = log_area_intersection / log_area_union
            with torch.no_grad():
                alpha_log = v / ((1 - iou_log) + v + 1e-16)
            ciou_log = iou_log - (center_distance / (c2 + 1e-16)) - alpha_log * v
            loss = 1 - ciou_log
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss
