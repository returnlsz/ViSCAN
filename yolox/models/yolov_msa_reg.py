#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import copy
import math
import time

import numpy
from loguru import logger
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from yolox.utils import bboxes_iou

from .losses import IOUloss
from .network_blocks import BaseConv, DWConv
from scipy.optimize import linear_sum_assignment
from yolox.utils.box_op import box_cxcywh_to_xyxy, generalized_box_iou
from yolox.models.post_trans import MSA_yolov
from torchvision.ops import roi_align
from yolox.models.post_process import postprocess
from yolox.models.post_process import my_postprocess
import numpy as np

class YOLOXHead(nn.Module):
    def __init__(
            self,
            num_classes,
            width=1.0,
            strides=[8, 16, 32],
            in_channels=[256, 512, 1024],
            act="silu",
            depthwise=False,
            heads=4,
            drop=0.0,
            use_score=True,
            defualt_p=20,
            sim_thresh=0.75,
            pre_nms=0.75,
            ave=True,
            defulat_pre=750,
            test_conf=0.001,
            use_mask=False
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()

        self.Afternum = defualt_p
        self.Prenum = defulat_pre
        self.simN = defualt_p
        self.nms_thresh = pre_nms
        self.n_anchors = 1
        self.use_score = use_score
        self.num_classes = num_classes
        self.decode_in_inference = True  # for deploy, set to False

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.cls_convs2 = nn.ModuleList()
        # reg
        self.reg_convs2 = nn.ModuleList()

        self.width = int(256 * width)
        self.trans = MSA_yolov(dim=self.width, out_dim=4 * self.width, num_heads=heads, attn_drop=drop)
        self.stems = nn.ModuleList()
        self.linear_pred = nn.Linear(int(4 * self.width),
                                     num_classes + 1)  # Mlp(in_features=512,hidden_features=self.num_classes+1)
        self.sim_thresh = sim_thresh
        self.ave = ave
        self.use_mask = use_mask
        Conv = DWConv if depthwise else BaseConv

        # reg
        # self.reg_linear_pred = nn.Linear(int(4 * self.width), 5)
        self.reg_linear_pred = nn.Linear(int(4 * self.width), 1)
        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.cls_convs2.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            # reg
            self.reg_convs2.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * 1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        
        for conv in self.reg_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin, labels=None, imgs=None, nms_thresh=0.5):
        outputs = []
        outputs_decode = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []
        before_nms_features = []
        before_nms_regf = []

        for k, (cls_conv, cls_conv2, reg_conv, reg_conv2,stride_this_level, x) in enumerate(
                zip(self.cls_convs, self.cls_convs2, self.reg_convs, self.reg_convs2,self.strides, xin)
        ):
            x = self.stems[k](x)  # 16X256X72X72
            reg_feat = reg_conv(x)  # 16X256X72X72
            reg_feat2 = reg_conv2(x)  # 16X256X72X72
            cls_feat = cls_conv(x)  # 16X256X72X72
            cls_feat2 = cls_conv2(x)  # 16X256X72X72

            # this part should be the same as the original model
            obj_output = self.obj_preds[k](reg_feat)  # 16X1X72X72
            reg_output = self.reg_preds[k](reg_feat)  # 16X4X72X72
            cls_output = self.cls_preds[k](cls_feat)  # 16X30X72X72
            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output], 1)
                output_decode = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
                )  # output的左上角的点和长宽由原来在特征图上的大小转化到原图的大小
                output, grid = self.get_output_and_grid(  # 得到每个格子的输出和左上角的坐标 ,还原到原图上（感觉是中心点）
                    output, k, stride_this_level, xin[0].type()
                )
                x_shifts.append(grid[:, :, 0])  # 得到每个格子的x轴坐标
                y_shifts.append(grid[:, :, 1])  # 得到每个格子的y轴坐标
                expanded_strides.append(  # 得到每个格子的步长
                    torch.zeros(1, grid.shape[1]).fill_(stride_this_level).type_as(xin[0])  # 填充stride_this_level步长
                )
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(
                        batch_size, self.n_anchors, 4, hsize, wsize
                    )
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    )
                    origin_preds.append(reg_output.clone())
                outputs.append(output)
                before_nms_features.append(cls_feat2)
                # before_nms_regf.append(reg_feat)
                before_nms_regf.append(reg_feat2)
            else:

                output_decode = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
                )

                # which features to choose
                before_nms_features.append(cls_feat2)  # 16X256X72X72
                # before_nms_regf.append(reg_feat)  # 16X256X72X72
                before_nms_regf.append(reg_feat2)  # 16X256X72X72
            outputs_decode.append(output_decode)  # 16X35X72X72

        self.hw = [x.shape[-2:] for x in outputs_decode]  # 72x72  36x36 18x18
        # output里的坐标已经发生了改变，还原到了原图的上，而output_decode的坐标还是原来的特征图上的大小，没有改变   坐标应该是中心点+长宽
        outputs_decode = torch.cat([x.flatten(start_dim=2) for x in outputs_decode], dim=2  # 16x6804x35
                                   ).permute(0, 2, 1)
        decode_res = self.decode_outputs(outputs_decode, dtype=xin[
            0].type())  # find topK predictions, play the same role as RPN  和上面的output操作差不多
        # nms   置信度 nms前30个   37（x1, y1, x2, y2, obj_conf, class_conf, class_pred,30）
        pred_result, pred_idx = self.postpro_woclass(decode_res, num_classes=self.num_classes, nms_thre=self.nms_thresh,
                                                     topK=self.Afternum)  # postprocess(decode_res,num_classes=30)
        # return pred_result
        if not self.training and imgs.shape[0] == 1:
            return self.postprocess_single_img(pred_result, self.num_classes)
        # YOLOv新的一条路
        cls_feat_flatten = torch.cat(  # 16x6804x256
            [x.flatten(start_dim=2) for x in before_nms_features], dim=2  # cls_feat2
        ).permute(0, 2, 1)  # [b,features,channels]
        reg_feat_flatten = torch.cat(  # 16x6804x256                      #reg_feat
            [x.flatten(start_dim=2) for x in before_nms_regf], dim=2
        ).permute(0, 2, 1)
        # 通过pred_idx找到对应cls、reg特征、cls_scores（类别分数） fg_scores（置信度 前后背景分数）    按照yolox得到的框的ID寻找
        features_cls, features_reg, cls_scores, fg_scores = self.find_feature_score(cls_feat_flatten, pred_idx,
                                                                                    reg_feat_flatten, imgs,
                                                                                    pred_result)
        features_reg = features_reg.unsqueeze(0)  # 1x240x256      240=30*batchsize(8)
        features_cls = features_cls.unsqueeze(0)  # 1x240x256
        if not self.training:
            cls_scores = cls_scores.to(cls_feat_flatten.dtype)
            fg_scores = fg_scores.to(cls_feat_flatten.dtype)
        if self.use_score:
            trans_cls, trans_reg = self.trans(features_cls, features_reg, cls_scores, fg_scores,
                                              sim_thresh=self.sim_thresh,
                                              ave=self.ave, use_mask=self.use_mask)
        else:
            trans_cls, trans_reg = self.trans(features_cls, features_reg, None, None, sim_thresh=self.sim_thresh,
                                              ave=self.ave)
        fc_output = self.linear_pred(trans_cls)
        # reg
        reg_fc_output = self.reg_linear_pred(trans_reg)

        fc_output = torch.reshape(fc_output, [outputs_decode.shape[0], -1, self.num_classes + 1])[:, :, :-1]

        # reg
        # reg_fc_output = torch.reshape(reg_fc_output, [outputs_decode.shape[0], -1, 5])

        reg_fc_output = torch.reshape(reg_fc_output, [outputs_decode.shape[0], -1, 1])

        reg_decode_res = reg_fc_output

        # 改变其位置，映射到原图，后续还要把更改的结果传入outputs 和 origin_pred和pred_res中
        # 根据pred_idx找到对应的位置
        # shape: batch_size * 30 * 5
        # reg_decode_res = self.sec_decode_outputs(pred_idx, reg_fc_output, dtype=xin[0].type())
        # # 得到的是xywh(不是比例)，还需要转成x1,y1,x2,y2
        # box_corner = reg_decode_res.new(reg_decode_res.shape)  # 中心点和长宽  -> 左上角右下角
        # box_corner[:, :, 0] = reg_decode_res[:, :, 0] - reg_decode_res[:, :, 2] / 2
        # box_corner[:, :, 1] = reg_decode_res[:, :, 1] - reg_decode_res[:, :, 3] / 2
        # box_corner[:, :, 2] = reg_decode_res[:, :, 0] + reg_decode_res[:, :, 2] / 2
        # box_corner[:, :, 3] = reg_decode_res[:, :, 1] + reg_decode_res[:, :, 3] / 2
        # reg_decode_res[:, :, :4] = box_corner[:, :, :4]  #


        # 将更改的结果传入outputs 和 origin_pred和pred_res中
        # orgin_ouput是根据尺度来索引的，先是72*72，36*36，18*18
        # if self.use_l1:
        #     for batch_idx in range(reg_decode_res.shape[0]):
        #         for ele_idx, ele in enumerate(pred_idx[batch_idx]):
        #             if ele <= 72 * 72 - 1:
        #                 origin_preds[0][batch_idx, ele, :] = reg_decode_res[batch_idx, ele_idx, :4]
        #             elif ele <= 72 * 72 + 36 * 36 - 1:
        #                 origin_preds[1][batch_idx, ele, :] = reg_decode_res[batch_idx, ele_idx, :4]
        #             else:
        #                 origin_preds[1][batch_idx, ele, :] = reg_decode_res[batch_idx, ele_idx, :4]

        # for batch_idx in range(reg_decode_res.shape[0]):
        #     pred_result[batch_idx][:, :4] = reg_decode_res[batch_idx, :, :4].squeeze(0)
        #     pred_result[batch_idx][:, 4] = reg_decode_res[batch_idx, :, 4].squeeze(0)
            # print("pred_res_shape:"+str(len(pred_result)))
        # print("reg_decode_res_shape:"+str(reg_decode_res.shape))
        # pred_result[:,:,:4] = reg_decode_res[...,:4].tolist()
        # pred_result[:,:, 4] = reg_decode_res[..., 4].sigmoid()

        # if self.training:
        #     return self.get_losses(
        #         imgs,
        #         x_shifts,
        #         y_shifts,
        #         expanded_strides,
        #         labels,
        #         torch.cat(outputs, 1),
        #         origin_preds,
        #         dtype=xin[0].dtype,
        #         refined_cls=fc_output,
        #         idx=pred_idx,
        #         pred_res=pred_result,
        #     )
        # 损失函数先不改，这里只是将每个batch最好的30个框的预测信息和obj进行了更新，没有改损失函数，为什么不改？应该改比较好，但是目前不会（）原来的损失函数是根据yolox预测出的batch_size * 6804 * 35进行
        # 正样本寻找，然后求损失函数+yolov的优化分类后的损失两部分。现在不改，因为reg特征聚合也用到了yolox的预测reg结果，如果yolox reg结果不好，那么也会影响到特征聚合的质量，所以先不改
        if self.training:
            return self.get_losses(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(outputs, 1),
                origin_preds,
                dtype=xin[0].dtype,
                refined_cls=fc_output,
                idx=pred_idx,
                pred_res=pred_result,
                refined_regs= reg_decode_res
            )
        else:

            class_conf, class_pred = torch.max(fc_output, -1, keepdim=False)  # 看哪个类别的概率最大
            result, result_ori = my_postprocess(copy.deepcopy(pred_result), self.num_classes, fc_output,reg_decode_res,
                                             nms_thre=nms_thresh)

            return result, result_ori  # result

    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])  # 用于生成网格坐标的
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid

        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)  # 增加一个anchor
        output = output.permute(0, 1, 3, 4, 2).reshape(  # 将 anchor*宽*高     batchx需要的anchor数x35
            batch_size, self.n_anchors * hsize * wsize, -1
        )
        grid = grid.view(1, -1, 2)  # （0,0） （0,1） （0,2） ...
        output[..., :2] = (output[..., :2] + grid) * stride  # 左上角的值   调整左上角点的位置
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride  # 长宽    调整长宽到图像的比例
        return output, grid

    def decode_outputs(self, outputs, dtype, flevel=0):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1,
                                                 2)  # 1x5184x2 https://blog.csdn.net/weixin_39504171/article/details/106074550
            grids.append(grid)  # grid形成网格 坐标（0,1）。。。。
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))  # 生成(*shape, 1), stride) 维stride   1x5184x1  里面的数字是stride

        # 现成 1* 6804 * 2
        grids = torch.cat(grids, dim=1).type(dtype)
        # 形成 1 * 6804 * 1
        strides = torch.cat(strides, dim=1).type(dtype)
        # output原形状16x6804x35
        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs

    def sec_decode_outputs(self, idx, outputs, dtype, flevel=0):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1,
                                                 2)  # 1x5184x2 https://blog.csdn.net/weixin_39504171/article/details/106074550
            grids.append(grid)  # grid形成网格 坐标（0,1）。。。。
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))  # 生成(*shape, 1), stride) 维stride   1x5184x1  里面的数字是stride

        # 现成 1* 6804 * 2
        grids = torch.cat(grids, dim=1).type(dtype)
        # 形成 1 * 6804 * 1
        strides = torch.cat(strides, dim=1).type(dtype)

        for batch_idx in range(outputs.shape[0]):
            for ele_idx, ele in enumerate(idx[batch_idx]):
                outputs[batch_idx, ele_idx, :2] = (outputs[batch_idx, ele_idx, :2] + grids[:, ele, :]) * strides[:, ele,
                                                                                                         :]
                outputs[batch_idx, ele_idx, 2:4] = torch.exp(outputs[batch_idx, ele_idx, 2:4]) * strides[:, ele, :]

        # output原形状16x6804x35
        # outputs[..., :2] = (outputs[..., :2] + grids) * strides
        # outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs

    def find_feature_score(self, features, idxs, reg_features, imgs=None, predictions=None, roi_features=None):
        features_cls = []
        features_reg = []
        cls_scores = []
        fg_scores = []
        for i, feature in enumerate(features):
            features_cls.append(feature[idxs[i][:self.simN]])  # 30x256
            features_reg.append(reg_features[i, idxs[i][:self.simN]])  # 30x256
            cls_scores.append(predictions[i][:self.simN, 5])  # 30x1   类别的分数
            fg_scores.append(predictions[i][:self.simN, 4])  # 30x1   置信度
        features_cls = torch.cat(features_cls)  # 240x256=30*8x256   30xbatch
        features_reg = torch.cat(features_reg)
        cls_scores = torch.cat(cls_scores)
        fg_scores = torch.cat(fg_scores)
        return features_cls, features_reg, cls_scores, fg_scores

    def get_losses(
            self,
            imgs,
            x_shifts,
            y_shifts,
            expanded_strides,
            labels,
            outputs,
            origin_preds,
            dtype,
            refined_cls,
            idx,
            pred_res,
            refined_regs
    ):
        # 原预测结果
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        # calculate targets  判断标签的类别是否是5个
        mixup = labels.shape[2] > 5
        if mixup:
            label_cut = labels[..., :5]
        else:
            label_cut = labels
        nlabel = (label_cut.sum(dim=2) > 0).sum(dim=1)  # number of objects    去除120中没有的标签 获取标签的真实个数

        total_num_anchors = outputs.shape[1]  # n_anchors_all ， 6804
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:  # 80轮之后的L1损失
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []
       
        ref_targets = []
        num_fg = 0.0
        num_gts = 0.0
        ref_masks = []

        # new_fg_masks = []
        # new_reg_targets = []
        new_obj_targets = []

        match_new_fg = 0
        for batch_idx in range(outputs.shape[0]):  # batch的大小
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
                ref_target = outputs.new_zeros((idx[batch_idx].shape[0], self.num_classes + 1))
                ref_target[:, -1] = 1

                new_obj_target = outputs.new_zeros((idx[batch_idx].shape[0], 1))
                # new_reg_target = outputs.new_zeros((0, 4))
                # new_fg_mask = outputs.new_zeros(idx[batch_idx].shape[0]).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]  # 每张图片的box
                gt_classes = labels[batch_idx, :num_gt, 0]  # 真实分类  [batch,120,class+xywh]
                bboxes_preds_per_image = bbox_preds[batch_idx]  # 每张图片的预测框

                try:
                    (
                        gt_matched_classes,  # 正样本的类别
                        fg_mask,  # 5376中正样本的mask掩码（位置信息）
                        pred_ious_this_matching,  # 正样本与它对应真实框的iou
                        matched_gt_inds,  # 正样本与真实框对应
                        num_fg_img,  # 正样本的数量
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                    )
                except RuntimeError:
                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                        "cpu",
                    )

                torch.cuda.empty_cache()  # 清除显存
                num_fg += num_fg_img

                cls_target_onehot = F.one_hot(  # 得到正样本对应的类别
                    gt_matched_classes.to(torch.int64), self.num_classes
                )
                cls_target = cls_target_onehot * pred_ious_this_matching.unsqueeze(-1)  # 使用iou作为其类别的分数（正样本与他对应真实框的iou）
                fg_idx = torch.where(fg_mask)[0]  # 30个正样本对应的ID

                obj_target = fg_mask.unsqueeze(-1)  # 置信度，位置的可信程度
                reg_target = gt_bboxes_per_image[matched_gt_inds]  # 框的目标 正样本对应的真实框
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

                # yolov独有
                ref_target = outputs.new_zeros((idx[batch_idx].shape[0], self.num_classes + 1))
                new_obj_target = outputs.new_zeros(idx[batch_idx].shape[0])
                # new_reg_target = outputs.new_zeros((0, 4))
                # new_fg_mask = outputs.new_zeros(idx[batch_idx].shape[0]).bool()
                fg = 0
                # -------------------------------------------------------------------------------------
                # 找到fg_mask中对应的正样本的idx，然后加入到new_reg_traget中,new_obj_target同理
                for ele_idx, ele in enumerate(idx[batch_idx]):  # 循环遍历预测框的索引
                    loc = torch.where(fg_idx == ele)[0]  # 正样本的ID与预测框扥ID相对应
                    if len(loc):
                        count = torch.sum(fg_mask[:ele+1])
                        match_new_fg += 1
                        # count = 0
                        # for i in range(ele+1):
                        #     if fg_mask[i]:
                        #         count += 1
                        # 当前元素对应第count-1个正样本
                        # new_reg_target = torch.cat((new_reg_target,gt_bboxes_per_image[matched_gt_inds[count-1]].unsqueeze(0)),dim = 0)
                        new_obj_target[ele_idx] = 1
                        # new_fg_mask[ele_idx] = True
                    else:
                        pass

                new_obj_target = new_obj_target.unsqueeze(-1)

                # -------------------------------------------------------------------------------------


            
                # 此处不知道要不要改？此处是根据原来的box选取与gt的iou，进而计算refined_cls的loss
                gt_xyxy = box_cxcywh_to_xyxy(torch.tensor(reg_target))
                pred_box = pred_res[batch_idx][:, :4]
                cost_giou, iou = generalized_box_iou(pred_box, gt_xyxy)  # 预测框与正样本对应的真实框对应的iou
                max_iou = torch.max(iou, dim=-1)  # 预测框对应最大的真实框iou
                for ele_idx, ele in enumerate(idx[batch_idx]):  # 循环遍历预测框的索引
                    loc = torch.where(fg_idx == ele)[0]  # 正样本的ID与预测框扥ID相对应
                    # 循环遍历预测框的ID看是否在正样本中 ，如果在就将对应的正样本与真实框的iou填入 对应得到mask就是0  如果不在 就执行下一步判断max_iou是否大于阈值，
                    # 大于把对应iou填入 mask=0  如果上两步都没有 则将 mask也就是最后一维改成1-max_iou
                    if len(loc):
                        ref_target[ele_idx, :self.num_classes] = cls_target[loc, :]
                        fg += 1
                        continue
                    if max_iou.values[ele_idx] >= 0.6:  # 预测框与真实框对应最大iou > 0.6
                        max_idx = int(max_iou.indices[ele_idx])

                        ref_target[ele_idx, :self.num_classes] = cls_target_onehot[max_idx, :] * max_iou.values[ele_idx]
                        fg += 1
                    else:
                        ref_target[ele_idx, -1] = 1 - max_iou.values[ele_idx]  # 最后一位记录的是1-max_iou的值
                # reg

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)

            # new_fg_masks.append(new_fg_mask)
            new_obj_targets.append(new_obj_target.to(dtype))
            # new_reg_targets.append(new_reg_target)
            # 这里主要是在yolox的基础上对类别进一步计算
            ref_targets.append(ref_target[:, :self.num_classes])
            ref_masks.append(ref_target[:, -1] == 0)  # 上面第31维值为0的为TRUE  也就是将上面类别分数过低的框删去 保留分较高的框
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = torch.cat(cls_targets, 0)  # 类别   30
        reg_targets = torch.cat(reg_targets, 0)  # 框的目标 正样本对应的真实框  4
        obj_targets = torch.cat(obj_targets, 0)  # 置信度 1
        ref_targets = torch.cat(ref_targets, 0)

        fg_masks = torch.cat(fg_masks, 0)  # 正样本对应的位置
        ref_masks = torch.cat(ref_masks, 0)

        # new_fg_masks = torch.cat(new_fg_masks, 0) 
        new_obj_targets = torch.cat(new_obj_targets, 0) 
        # new_reg_targets = torch.cat(new_reg_targets, 0) 
        # print(sum(ref_masks)/ref_masks.shape[0])
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)
        loss_iou = (  # 预测框与正样本对应的真实框做iou_loss
                       self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
                   ).sum() / num_fg

        # loss_obj = (  # 预测的置信度与正样本的位置信息的做loss
        #                self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
        #            ).sum() / num_fg
        
        loss_obj = (
            self.focal_loss(obj_preds.sigmoid().view(-1, 1), obj_targets)
        ).sum() / num_fg
        
        loss_cls = (  # 预测的类别与正样本类别做loss
                       self.bcewithlog_loss(
                           cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
                       )
                   ).sum() / num_fg
        loss_ref = (
                       self.bcewithlog_loss(
                           refined_cls.view(-1, self.num_classes)[ref_masks], ref_targets[ref_masks]
                       )
                   ).sum() / num_fg

        # ---------------------------------------------------------
        # new_bbox_preds = refined_regs[:,:,:4]
        # new_loss_iou = (
        #     self.iou_loss(new_bbox_preds.view(-1,4)[new_fg_masks], new_reg_targets)
        # ).sum() / num_fg
        new_obj_preds = refined_regs[:,:,0]
        new_loss_obj = (  
                       self.focal_loss(new_obj_preds.sigmoid().view(-1, 1), new_obj_targets)
                   ).sum() / match_new_fg

        # ---------------------------------------------------------
        if self.use_l1:
            loss_l1 = (
                          self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
                      ).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 3.0

        # print("loss_iou:"+str(3 * loss_iou))
        # print("loss_obj:"+str(loss_obj))
        # print("loss_ref:"+str(2 * loss_ref))
        # print("loss_l1:"+str(loss_l1))
        # print("loss_cls:"+str(loss_cls))
        # print("new_loss_iou:"+str(reg_weight * new_loss_iou))
        # print("new_loss_obj:"+str(new_loss_obj))
        # loss = reg_weight * loss_iou + loss_obj + 2 * loss_ref + loss_l1 + loss_cls + reg_weight * new_loss_iou + new_loss_obj
        loss = reg_weight * loss_iou + loss_obj + 2 * loss_ref + loss_l1 + loss_cls + new_loss_obj
        return (
            loss,
            reg_weight * loss_iou,
            loss_l1,
            loss_obj,
            2 * loss_ref,
            new_loss_obj,
            loss_cls,
            num_fg / max(num_gts, 1),
        )
    
    def focal_loss(self, pred, gt):
        pos_inds = gt.eq(1).float()
        neg_inds = gt.eq(0).float()
        pos_loss = torch.log(pred+1e-5) * torch.pow(1 - pred, 2) * pos_inds * 0.75
        neg_loss = torch.log(1 - pred+1e-5) * torch.pow(pred, 2) * neg_inds * 0.25
        loss = -(pos_loss + neg_loss)
        return loss
    
    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(
            self,
            batch_idx,
            num_gt,
            total_num_anchors,
            gt_bboxes_per_image,
            gt_classes,
            bboxes_preds_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            cls_preds,
            bbox_preds,
            obj_preds,
            labels,
            imgs,
            mode="gpu",
    ):

        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()
        # 预测框的中心点既在真实框中也在4.5x4.5中的预测框
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
        )
        # 根据是否在框中，删除不在框中的数据
        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)  # 计算真实框和预测框的iou

        gt_cls_per_image = (  # 4x656x30 一张图片上有四个真实框 每个框的类别复制656 对应着656个预测框
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
                .float()
                .unsqueeze(1)
                .repeat(1, num_in_boxes_anchor, 1)
        )
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (  # 置信度*类别
                    cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()  # 4x656x30    656x30 复制4份
                    * obj_preds_.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()  # 4x656x1
            )
            pair_wise_cls_loss = F.binary_cross_entropy(  # 预测框的类别和真实框的类别做计算 得出他的类别
                cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
            ).sum(-1)
        del cls_preds_

        cost = (
                pair_wise_cls_loss  # 分类的一个损失
                + 3.0 * pair_wise_ious_loss  # iou损失
                + 100000.0 * (~is_in_boxes_and_center)  # 如果不在里面 给她一个很大的值，cos就不会选到他
        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_in_boxes_info(
            self,
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
    ):
        expanded_strides_per_image = expanded_strides[0]
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image  # 左上角的坐标
        x_centers_per_image = (  # 计算每一个格子的中心点的位置
            (x_shifts_per_image + 0.5 * expanded_strides_per_image)
                .unsqueeze(0)
                .repeat(num_gt, 1)
        )  # [n_anchor] -> [n_gt, n_anchor]
        y_centers_per_image = (
            (y_shifts_per_image + 0.5 * expanded_strides_per_image)  # 每个格子的中心点
                .unsqueeze(0)
                .repeat(num_gt, 1)
        )
        # 计算真实框的四边  l_x l_y r_x r_y
        gt_bboxes_per_image_l = (
            (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])  # 中心点和长宽
                .unsqueeze(1)
                .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_r = (
            (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
                .unsqueeze(1)
                .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_t = (
            (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
                .unsqueeze(1)
                .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_b = (
            (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
                .unsqueeze(1)
                .repeat(1, total_num_anchors)
        )
        # 判断5376个框那些中心点在真实框中
        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)
        # 4x5376x4  四个真实框 5376个预测框 四个xy相减的值   -》4x5376 mask
        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0  # 四个真实框都没有的预测框去除（这里真实框的数量为4，可能不同）
        # in fixed center
        # 与上面一样
        center_radius = 4.5  # 生成一个4.5x4.5的格子

        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all  # 两者并集   预测框或者在真实框或者在4.5x4.5中

        is_in_boxes_and_center = (
                is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]  # 两者交集
        )
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # Dynamic K
        # ---------------------------------------------------------------
        matching_matrix = torch.zeros_like(cost)

        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))  # 取10个iou计算
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k,
                                  dim=1)  # 选取不大余10个iou 4x652 -> 4x10 取每一行的前十个：每一个真实框对应的前十个预测框的iou
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)  # iou求和向下取整 正样本的个数
        for gt_idx in range(num_gt):  # 主要是取出659个中cost最小的前几个 这个几个是前面 iou相加得到的  最小值也就是对应着正样本
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False  # 按照最小排序 ，cos越小越好   取出cost里前k个最小的id
            )
            matching_matrix[gt_idx][pos_idx] = 1.0  # 将其对应的位置变成1

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)  # 以防一个格子有两个真实框
        if (anchor_matching_gt > 1).sum() > 0:  # 是否又大于1的情况，有的话选取cos最小的作为正样本
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0.0  # 大于1全为0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0  # cos最小的改为1
        fg_mask_inboxes = matching_matrix.sum(0) > 0.0  # 659样本选择区间哪些被改成正样本
        num_fg = fg_mask_inboxes.sum().item()  # 正样本个数

        fg_mask[fg_mask.clone()] = fg_mask_inboxes  # 总框数里面哪些是真样本

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)  # 返回最大的位置，也就是1对应的真实框的位置 每个正样本对应的真实框
        gt_matched_classes = gt_classes[matched_gt_inds]  # 每个正样本对应的真实类别

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[  # 每个正样本与真实框对应的iou
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds

    def postpro_woclass(self, prediction, num_classes, nms_thre=0.75, topK=75, features=None):
        # find topK predictions, play the same role as RPN
        '''

        Args:
            prediction: [batch,feature_num,5+clsnum]
            num_classes:
            conf_thre:
            conf_thre_high:
            nms_thre:

        Returns:
            [batch,topK,5+clsnum]
        '''
        
        self.topK = topK

        low = 10
        high = self.topK - low

        box_corner = prediction.new(prediction.shape)  # 中心点和长宽  -> 左上角右下角
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]  #
        output = [None for _ in range(len(prediction))]
        output_index = [None for _ in range(len(prediction))]
        features_list = []
        for i, image_pred in enumerate(prediction):

            if not image_pred.size(0):
                continue
            # Get score and class with highest confidence
            class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1,
                                               keepdim=True)  # 返回每一个batch（每一帧）30个类别每一行最大值并返回索引

            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            detections = torch.cat(
                (image_pred[:, :5], class_conf, class_pred.float(), image_pred[:, 5: 5 + num_classes]), 1)
            # 获取置信度前750个
            conf_score = image_pred[:, 4]
            top_pre = torch.topk(conf_score, k=self.Prenum)
            sort_idx = top_pre.indices[:self.Prenum]
            detections_temp = detections[sort_idx, :]
            # 根据每个类别进行过滤，只对同一种类别进行计算IOU和阈值过滤。
            # boxes: Tensor, 预测框
            # scores: Tensor, 预测置信度
            # idxs: Tensor, 预测框类别
            # iou_threshold: float, IOU阈值
            nms_out_index = torchvision.ops.batched_nms(
                detections_temp[:, :4],  # x1, y1, x2, y2,
                detections_temp[:, 4] * detections_temp[:, 5],  # obj_conf * class_conf   目标的置信度*类别的置信度
                detections_temp[:, 6],  # class_pred   预测的类别
                nms_thre,
            )

            topk_idx = sort_idx[nms_out_index[:self.topK]]  # nms最大的前30个
            # topk_idx = torch.cat((topk_idx,sort_idx[nms_out_index[-1*low:]]),dim=0)
            
            output[i] = detections[topk_idx, :]
            output_index[i] = topk_idx

        return output, output_index

    def postprocess_single_img(self, prediction, num_classes, conf_thre=0.001, nms_thre=0.5):

        output_ori = [None for _ in range(len(prediction))]
        prediction_ori = copy.deepcopy(prediction)
        for i, detections in enumerate(prediction):

            if not detections.size(0):
                continue

            detections_ori = prediction_ori[i]

            conf_mask = (detections_ori[:, 4] * detections_ori[:, 5] >= conf_thre).squeeze()
            detections_ori = detections_ori[conf_mask]
            nms_out_index = torchvision.ops.batched_nms(
                detections_ori[:, :4],
                detections_ori[:, 4] * detections_ori[:, 5],
                detections_ori[:, 6],
                nms_thre,
            )
            detections_ori = detections_ori[nms_out_index]
            output_ori[i] = detections_ori
        # print(output)
        return output_ori, output_ori
