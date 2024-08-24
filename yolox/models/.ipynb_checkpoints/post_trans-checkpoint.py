import copy

import torch
import torch.nn as nn
from .weight_init import trunc_normal_
from .losses import IOUloss
from torch.nn import functional as F
from matplotlib import pyplot as plt
from yolox.utils.box_op import box_cxcywh_to_xyxy, generalized_box_iou

def visual_attention(data):
    data = data.cpu()
    data = data.detach().numpy()

    plt.xlabel('x')
    plt.ylabel('score')
    plt.imshow(data)
    plt.show()

# class Attention_msa(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., scale=25):
#         # dim :input[batchsize,sequence length, input dimension]-->output[batchsize, sequence lenght, dim]
#         # qkv_bias : Is it matter?
#         # qk_scale, attn_drop,proj_drop will not be used
#         # object = Attention(dim,num head)
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
#         self.scale = scale  # qk_scale or head_dim ** -0.5

#         self.qkv_cls = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.qkv_reg = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)

#     def forward(self, x_cls, x_reg, cls_score=None, fg_score=None, return_attention=False, ave=True, sim_thresh=0.75,
#                 use_mask=False):
#         B, N, C = x_cls.shape
#         # 1x240x768(256*3) -> 1x240x3x4x64  ->3x1x4x240x64  3：qkv 4:四个头 240:240个
#         # 简单理解就是由原来的一个1x240x256经过全连接变成了3个，代表qkv 把256分成四份，交由四个头计算，后面再融合(不一定准确)
#         #q:我要找个零食吃 k:1 v:饼干   k:2  v:薯片
#         qkv_cls = self.qkv_cls(x_cls).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,4)  # 3, B, num_head, N, c
#         qkv_reg = self.qkv_reg(x_reg).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q_cls, k_cls, v_cls = qkv_cls[0], qkv_cls[1], qkv_cls[2]  # make torchscript happy (cannot use tensor as tuple)
#         q_reg, k_reg, v_reg = qkv_reg[0], qkv_reg[1], qkv_reg[2]
#         #归一化
#         q_cls = q_cls / torch.norm(q_cls, dim=-1, keepdim=True) #最后一个维度求2的范数 维度保持不变
#         k_cls = k_cls / torch.norm(k_cls, dim=-1, keepdim=True)
#         q_reg = q_reg / torch.norm(q_reg, dim=-1, keepdim=True)
#         k_reg = k_reg / torch.norm(k_reg, dim=-1, keepdim=True)
#         v_cls_normed = v_cls / torch.norm(v_cls, dim=-1, keepdim=True)

#         # reg
#         v_reg_normed = v_reg / torch.norm(v_reg, dim=-1, keepdim=True)

#         if cls_score == None:
#             cls_score = 1
#         else:  # 将其与权重相成，相当于给分高的更多的权重
#             cls_score = torch.reshape(cls_score, [1, 1, 1, -1]).repeat(1, self.num_heads, N, 1) # 哪个维度多少，哪个维度复制多少

#         if fg_score == None:
#             fg_score = 1
#         else:    # 240-> 1x1x1x240-> 1x4x240x240
#             fg_score = torch.reshape(fg_score, [1, 1, 1, -1]).repeat(1, self.num_heads, N, 1)

#         attn_cls_raw = v_cls_normed @ v_cls_normed.transpose(-2, -1)

#         # reg
#         attn_reg_raw = v_reg_normed @ v_reg_normed.transpose(-2, -1)

#         if use_mask:
#             # only reference object with higher confidence..
#             cls_score_mask = (cls_score > (cls_score.transpose(-2, -1) - 0.1)).type_as(cls_score)
#             fg_score_mask = (fg_score > (fg_score.transpose(-2, -1) - 0.1)).type_as(fg_score)
#         else:
#             cls_score_mask = fg_score_mask = 1

#         # cls_score_mask = (cls_score < (cls_score.transpose(-2, -1) + 0.1)).type_as(cls_score)
#         # fg_score_mask = (fg_score < (fg_score.transpose(-2, -1) + 0.1)).type_as(fg_score)
#         # visual_attention(cls_score[0, 0, :, :])
#         # visual_attention(cls_score_mask[0,0,:,:])

#         attn_cls = (q_cls @ k_cls.transpose(-2, -1)) * self.scale * cls_score * cls_score_mask  # 分少的给更少的权重
#         attn_cls = attn_cls.softmax(dim=-1)
#         attn_cls = self.attn_drop(attn_cls)   # 注意力的权重

#         attn_reg = (q_reg @ k_reg.transpose(-2, -1)) * self.scale * fg_score * fg_score_mask
#         attn_reg = attn_reg.softmax(dim=-1)
#         attn_reg = self.attn_drop(attn_reg)

#         attn = (attn_reg + attn_cls) / 2
#         x = (attn @ v_cls).transpose(1, 2).reshape(B, N, C)  # 1x4x240x64 -> 1x240x256 经过注意力机制后输出的特征

#         x_ori = v_cls.permute(0, 2, 1, 3).reshape(B, N, C)
#         x_cls = torch.cat([x, x_ori], dim=-1)   # 输出特征和v的融合 有点像残差网络

#         # reg
#         reg_x = (attn @ v_reg).transpose(1, 2).reshape(B, N, C)  # 1x4x240x64 -> 1x240x256 经过注意力机制后输出的特征

#         reg_x_ori = v_reg.permute(0, 2, 1, 3).reshape(B, N, C)
#         x_reg = torch.cat([reg_x, reg_x_ori], dim=-1)  # 输出特征和v的融合 有点像残差网络

#         if ave:
#             ones_matrix = torch.ones(attn.shape[2:]).to('cuda')
#             zero_matrix = torch.zeros(attn.shape[2:]).to('cuda')
#             #sim_mask 自己和自己的权重肯定大，
#             attn_cls_raw = torch.sum(attn_cls_raw, dim=1, keepdim=False)[0] / self.num_heads  # 多头特征的平均
#             sim_mask = torch.where(attn_cls_raw > sim_thresh, ones_matrix, zero_matrix)       # 应该是设置阈值过滤低于0.75的值
#             sim_attn = torch.sum(attn, dim=1, keepdim=False)[0] / self.num_heads              # 多头权重的平均

#             sim_round2 = torch.softmax(sim_attn, dim=-1)  # 多头权重的平均 softmax
#             sim_round2 = sim_mask * sim_round2 / (torch.sum(sim_mask * sim_round2, dim=-1, keepdim=True))#相当于sim_mask * sim_round2的softmax

#             # reg
#             attn_reg_raw = torch.sum(attn_reg_raw, dim=1, keepdim=False)[0] / self.num_heads  # 多头特征的平均
#             reg_sim_mask = torch.where(attn_reg_raw > sim_thresh, ones_matrix, zero_matrix)  # 应该是设置阈值过滤低于0.75的值
#             reg_sim_attn = torch.sum(attn, dim=1, keepdim=False)[0] / self.num_heads  # 多头权重的平均

#             reg_sim_round2 = torch.softmax(reg_sim_attn, dim=-1)  # 多头权重的平均 softmax
#             reg_sim_round2 = reg_sim_mask * reg_sim_round2 / (
#             torch.sum(reg_sim_mask * reg_sim_round2, dim=-1, keepdim=True))  # 相当于sim_mask * sim_round2的softmax
#             return x_cls, x_reg, sim_round2,reg_sim_round2
#         else:
#             return x_cls, x_reg, None,None
        
class Attention_msa(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., scale=25):
        # dim :input[batchsize,sequence length, input dimension]-->output[batchsize, sequence lenght, dim]
        # qkv_bias : Is it matter?
        # qk_scale, attn_drop,proj_drop will not be used
        # object = Attention(dim,num head)
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = scale  # qk_scale or head_dim ** -0.5

        self.qkv_cls = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_reg = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x_cls, x_reg, cls_score=None, fg_score=None, return_attention=False, ave=True, sim_thresh=0.75,
                use_mask=False):
        B, N, C = x_cls.shape

        qkv_cls = self.qkv_cls(x_cls).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                                    4)  # 3, B, num_head, N, c
        qkv_reg = self.qkv_reg(x_reg).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_cls, k_cls, v_cls = qkv_cls[0], qkv_cls[1], qkv_cls[2]  # make torchscript happy (cannot use tensor as tuple)
        q_reg, k_reg, v_reg = qkv_reg[0], qkv_reg[1], qkv_reg[2]

        q_cls = q_cls / torch.norm(q_cls, dim=-1, keepdim=True)
        k_cls = k_cls / torch.norm(k_cls, dim=-1, keepdim=True)
        q_reg = q_reg / torch.norm(q_reg, dim=-1, keepdim=True)
        k_reg = k_reg / torch.norm(k_reg, dim=-1, keepdim=True)
        v_cls_normed = v_cls / torch.norm(v_cls, dim=-1, keepdim=True)

        if cls_score == None:
            cls_score = 1
        else:
            cls_score = torch.reshape(cls_score, [1, 1, 1, -1]).repeat(1, self.num_heads, N, 1)

        if fg_score == None:
            fg_score = 1
        else:
            fg_score = torch.reshape(fg_score, [1, 1, 1, -1]).repeat(1, self.num_heads, N, 1)

        attn_cls_raw = v_cls_normed @ v_cls_normed.transpose(-2, -1)
        if use_mask:
            # only reference object with higher confidence..
            cls_score_mask = (cls_score > (cls_score.transpose(-2, -1) - 0.1)).type_as(cls_score)
            fg_score_mask = (fg_score > (fg_score.transpose(-2, -1) - 0.1)).type_as(fg_score)
        else:
            cls_score_mask = fg_score_mask = 1

        # cls_score_mask = (cls_score < (cls_score.transpose(-2, -1) + 0.1)).type_as(cls_score)
        # fg_score_mask = (fg_score < (fg_score.transpose(-2, -1) + 0.1)).type_as(fg_score)
        # visual_attention(cls_score[0, 0, :, :])
        # visual_attention(cls_score_mask[0,0,:,:])

        attn_cls = (q_cls @ k_cls.transpose(-2, -1)) * self.scale * cls_score * cls_score_mask
        attn_cls = attn_cls.softmax(dim=-1)
        attn_cls = self.attn_drop(attn_cls)

        attn_reg = (q_reg @ k_reg.transpose(-2, -1)) * self.scale * fg_score * fg_score_mask
        attn_reg = attn_reg.softmax(dim=-1)
        attn_reg = self.attn_drop(attn_reg)

        attn = (attn_reg + attn_cls) / 2
        x = (attn @ v_cls).transpose(1, 2).reshape(B, N, C)

        x_ori = v_cls.permute(0, 2, 1, 3).reshape(B, N, C)
        x_cls = torch.cat([x, x_ori], dim=-1)
        #

        if ave:
            ones_matrix = torch.ones(attn.shape[2:]).to('cuda')
            zero_matrix = torch.zeros(attn.shape[2:]).to('cuda')

            attn_cls_raw = torch.sum(attn_cls_raw, dim=1, keepdim=False)[0] / self.num_heads
            sim_mask = torch.where(attn_cls_raw > sim_thresh, ones_matrix, zero_matrix)
            sim_attn = torch.sum(attn, dim=1, keepdim=False)[0] / self.num_heads

            sim_round2 = torch.softmax(sim_attn, dim=-1)
            sim_round2 = sim_mask * sim_round2 / (torch.sum(sim_mask * sim_round2, dim=-1, keepdim=True))
            return x_cls, None, sim_round2
        else:
            return x_cls, None, None

class Attention_msa_visual(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False,attn_drop=0., scale=25):
        # dim :input[batchsize,sequence length, input dimension]-->output[batchsize, sequence lenght, dim]
        # qkv_bias : Is it matter?
        # qk_scale, attn_drop,proj_drop will not be used
        # object = Attention(dim,num head)
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = 30#scale  # qk_scale or head_dim ** -0.5

        self.qkv_cls = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_reg = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x_cls, x_reg, cls_score=None, fg_score=None,img = None, pred = None):
        B, N, C = x_cls.shape

        qkv_cls = self.qkv_cls(x_cls).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                                    4)  # 3, B, num_head, N, c
        qkv_reg = self.qkv_reg(x_reg).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_cls, k_cls, v_cls = qkv_cls[0], qkv_cls[1], qkv_cls[2]  # make torchscript happy (cannot use tensor as tuple)
        q_reg, k_reg, v_reg = qkv_reg[0], qkv_reg[1], qkv_reg[2]

        q_cls = q_cls / torch.norm(q_cls, dim=-1, keepdim=True)
        k_cls = k_cls / torch.norm(k_cls, dim=-1, keepdim=True)
        q_reg = q_reg / torch.norm(q_reg, dim=-1, keepdim=True)
        k_reg = k_reg / torch.norm(k_reg, dim=-1, keepdim=True)
        v_cls_normed = v_cls / torch.norm(v_cls,dim=-1,keepdim=True)

        if cls_score == None:
            cls_score = 1
        else:
            cls_score = torch.reshape(cls_score,[1,1,1,-1]).repeat(1,self.num_heads,N, 1)

        if fg_score == None:
            fg_score = 1
        else:
            fg_score = torch.reshape(fg_score, [1, 1, 1, -1]).repeat(1,self.num_heads,N, 1)

        attn_cls_raw = v_cls_normed @ v_cls_normed.transpose(-2, -1)

        attn_cls = (q_cls @ k_cls.transpose(-2, -1)) * self.scale * cls_score #* cls_score
        attn_cls = attn_cls.softmax(dim=-1)
        attn_cls = self.attn_drop(attn_cls)

        attn_reg = (q_reg @ k_reg.transpose(-2, -1)) * self.scale * fg_score
        attn_reg = attn_reg.softmax(dim=-1)
        attn_reg = self.attn_drop(attn_reg)

        attn = (attn_cls_raw*25).softmax(dim=-1)#attn_cls#(attn_reg + attn_cls) / 2 #attn_reg#(attn_reg + attn_cls) / 2
        x = (attn @ v_cls).transpose(1, 2).reshape(B, N, C)

        x_ori = v_cls.permute(0,2,1,3).reshape(B, N, C)
        x_cls = torch.cat([x, x_ori], dim=-1)

        ones_matrix = torch.ones(attn.shape[2:]).to('cuda')
        zero_matrix = torch.zeros(attn.shape[2:]).to('cuda')

        attn_cls_raw = torch.sum(attn_cls_raw,dim=1,keepdim=False)[0] / self.num_heads
        sim_mask = torch.where(attn_cls_raw > 0.75, ones_matrix, zero_matrix)
        sim_attn = torch.sum(attn, dim=1, keepdim=False)[0] / self.num_heads

        sim_round2 = torch.softmax(sim_attn, dim=-1)
        sim_round2 = sim_mask*sim_round2/(torch.sum(sim_mask*sim_round2,dim=-1,keepdim=True))
        from yolox.models.post_process import visual_sim
        attn_total = torch.sum(attn,dim=1,keepdim=False)[0] / self.num_heads
        visual_sim(attn_total,img,30,pred,attn_cls_raw)
        return x_cls,None,sim_round2



class Attention_msa_online(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False,attn_drop=0., scale=25):
        # dim :input[batchsize,sequence length, input dimension]-->output[batchsize, sequence lenght, dim]
        # qkv_bias : Is it matter?
        # qk_scale, attn_drop,proj_drop will not be used
        # object = Attention(dim,num head)
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = scale  # qk_scale or head_dim ** -0.5
        self.qkv_cls = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_reg = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x_cls, x_reg, cls_score=None, fg_score=None, return_attention=False,ave = True):
        B, N, C = x_cls.shape

        qkv_cls = self.qkv_cls(x_cls).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                                    4)  # 3, B, num_head, N, c
        qkv_reg = self.qkv_reg(x_reg).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_cls, k_cls, v_cls = qkv_cls[0], qkv_cls[1], qkv_cls[2]  # make torchscript happy (cannot use tensor as tuple)
        q_reg, k_reg, v_reg = qkv_reg[0], qkv_reg[1], qkv_reg[2]

        q_cls = q_cls / torch.norm(q_cls, dim=-1, keepdim=True)
        k_cls = k_cls / torch.norm(k_cls, dim=-1, keepdim=True)
        q_reg = q_reg / torch.norm(q_reg, dim=-1, keepdim=True)
        k_reg = k_reg / torch.norm(k_reg, dim=-1, keepdim=True)
        v_cls_normed = v_cls / torch.norm(v_cls,dim=-1,keepdim=True)

        if cls_score == None:
            cls_score = 1
        else:
            cls_score = torch.reshape(cls_score,[1,1,1,-1]).repeat(1,self.num_heads,N, 1)

        if fg_score == None:
            fg_score = 1
        else:
            fg_score = torch.reshape(fg_score, [1, 1, 1, -1]).repeat(1,self.num_heads,N, 1)

        attn_cls_raw = v_cls_normed @ v_cls_normed.transpose(-2, -1)

        attn_cls = (q_cls @ k_cls.transpose(-2, -1)) * self.scale * cls_score
        attn_cls = attn_cls.softmax(dim=-1)
        attn_cls = self.attn_drop(attn_cls)

        attn_reg = (q_reg @ k_reg.transpose(-2, -1)) * self.scale * fg_score
        attn_reg = attn_reg.softmax(dim=-1)
        attn_reg = self.attn_drop(attn_reg)

        attn = (attn_reg + attn_cls) / 2
        x = (attn @ v_cls).transpose(1, 2).reshape(B, N, C)

        x_ori = v_cls.permute(0,2,1,3).reshape(B, N, C)
        x_cls = torch.cat([x, x_ori], dim=-1)
        if ave:
            ones_matrix = torch.ones(attn.shape[2:]).to('cuda')
            zero_matrix = torch.zeros(attn.shape[2:]).to('cuda')

            attn_cls_raw = torch.sum(attn_cls_raw,dim=1,keepdim=False)[0] / self.num_heads
            sim_mask = torch.where(attn_cls_raw > 0.75, ones_matrix, zero_matrix)
            sim_attn = torch.sum(attn, dim=1, keepdim=False)[0] / self.num_heads

            sim_round2 = torch.softmax(sim_attn, dim=-1)
            sim_round2 = sim_mask*sim_round2/(torch.sum(sim_mask*sim_round2,dim=-1,keepdim=True))
            return x_cls,None,sim_round2
        else:
            return x_cls

# class MSA_yolov(nn.Module):

#     def __init__(self, dim, out_dim, num_heads=4, qkv_bias=False, attn_drop=0., scale=25):
#         super().__init__()
#         self.msa = Attention_msa(dim, num_heads, qkv_bias, attn_drop, scale=scale)
#         self.linear1 = nn.Linear(2 * dim, 2 * dim)
#         self.linear2 = nn.Linear(4 * dim, out_dim)

#     def find_similar_round2(self, features, sort_results):
#         key_feature = features[0]
#         support_feature = features[0]
#         if not self.training:
#             sort_results = sort_results.to(features.dtype)
#         soft_sim_feature = (
#                     sort_results @ support_feature)  # .transpose(1, 2)#torch.sum(softmax_value * most_sim_feature, dim=1)
#         cls_feature = torch.cat([soft_sim_feature, key_feature], dim=-1)
#         return cls_feature

#     def forward(self, x_cls, x_reg, cls_score=None, fg_score=None, sim_thresh=0.75, ave=True, use_mask=False):
#         trans_cls, trans_reg, sim_round2 = self.msa(x_cls, x_reg, cls_score, fg_score, sim_thresh=sim_thresh, ave=ave,
#                                                     use_mask=use_mask)
#         msa = self.linear1(trans_cls)
#         msa = self.find_similar_round2(msa, sim_round2)

#         out = self.linear2(msa)
#         return out

class MSA_yolov(nn.Module):

    def __init__(self, dim, out_dim, num_heads=4, qkv_bias=False, attn_drop=0., scale=25):
        super().__init__()
        self.msa = Attention_msa(dim, num_heads, qkv_bias, attn_drop, scale=scale)
        self.linear1 = nn.Linear(2 * dim, 2 * dim)
        # self.linear2 = nn.Linear(4 * dim, out_dim)
        self.linear2 = nn.Linear(4 * dim, 4 * dim)

        # reg
        self.reg_linear1 = nn.Linear(2 * dim, 2 * dim)
        self.reg_linear2 = nn.Linear(4 * dim, 4 * dim)

    def find_similar_round2(self, features, sort_results,reg_features,reg_sort_results):
        key_feature = features[0]
        support_feature = features[0]
        if not self.training:
            sort_results = sort_results.to(features.dtype)
        soft_sim_feature = (
                    sort_results @ support_feature)  # .transpose(1, 2)#torch.sum(softmax_value * most_sim_feature, dim=1)
        cls_feature = torch.cat([soft_sim_feature, key_feature], dim=-1)

        reg_key_feature = reg_features[0]
        reg_support_feature = reg_features[0]
        if not self.training:
            reg_sort_results = reg_sort_results.to(features.dtype)
        reg_soft_sim_feature = (
                reg_sort_results @ reg_support_feature)  # .transpose(1, 2)#torch.sum(softmax_value * most_sim_feature, dim=1)
        reg_feature = torch.cat([reg_soft_sim_feature, reg_key_feature], dim=-1)
        return cls_feature,reg_feature

    def forward(self, x_cls, x_reg, cls_score=None, fg_score=None, sim_thresh=0.75, ave=True, use_mask=False):
        trans_cls, trans_reg, sim_round2,reg_sim_round2 = self.msa(x_cls, x_reg, cls_score, fg_score, sim_thresh=sim_thresh, ave=ave,
                                                    use_mask=use_mask)
        msa = self.linear1(trans_cls)
        reg_msa = self.reg_linear1(trans_reg)
        msa,reg_msa = self.find_similar_round2(msa, sim_round2,reg_msa,reg_sim_round2)

        out = self.linear2(msa)
        reg_out = self.reg_linear2(reg_msa)
        return out,reg_out

class MSA_yolov2(nn.Module):

    def __init__(self, dim, out_dim, num_heads=4, qkv_bias=False, attn_drop=0., scale=25):
        super().__init__()
        self.msa = Attention_msa(dim, num_heads, qkv_bias, attn_drop, scale=scale)
        self.msa2 = Attention_msa(dim, num_heads, qkv_bias, attn_drop, scale=scale)
        self.linear1 = nn.Linear(2 * dim, 2 * dim)
        # self.linear2 = nn.Linear(4 * dim, out_dim)
        self.linear2 = nn.Linear(4 * dim, 4 * dim)

        # reg
        self.reg_linear1 = nn.Linear(2 * dim, 2 * dim)
        self.reg_linear2 = nn.Linear(4 * dim, 4 * dim)

    def find_similar_round2(self, features, sort_results,reg_features,reg_sort_results):
        key_feature = features[0]
        support_feature = features[0]
        if not self.training:
            sort_results = sort_results.to(features.dtype)
        soft_sim_feature = (
                    sort_results @ support_feature)  # .transpose(1, 2)#torch.sum(softmax_value * most_sim_feature, dim=1)
        cls_feature = torch.cat([soft_sim_feature, key_feature], dim=-1)

        reg_key_feature = reg_features[0]
        reg_support_feature = reg_features[0]
        if not self.training:
            reg_sort_results = reg_sort_results.to(features.dtype)
        reg_soft_sim_feature = (
                reg_sort_results @ reg_support_feature)  # .transpose(1, 2)#torch.sum(softmax_value * most_sim_feature, dim=1)
        reg_feature = torch.cat([reg_soft_sim_feature, reg_key_feature], dim=-1)
        return cls_feature,reg_feature

    def forward(self, x_cls, x_reg, x_cls2, x_reg2, cls_score=None, fg_score=None, sim_thresh=0.75, ave=True, use_mask=False):
        trans_cls, None_value, sim_round2 = self.msa(x_cls, x_reg2, cls_score, fg_score, sim_thresh=sim_thresh, ave=ave,
                                                    use_mask=use_mask)
        trans_reg, None_value, reg_sim_round2 = self.msa2(x_reg, x_cls2, fg_score, cls_score, sim_thresh=sim_thresh, ave=ave,
                                                    use_mask=use_mask)
        msa = self.linear1(trans_cls)
        reg_msa = self.reg_linear1(trans_reg)
        msa,reg_msa = self.find_similar_round2(msa, sim_round2,reg_msa,reg_sim_round2)

        out = self.linear2(msa)
        reg_out = self.reg_linear2(reg_msa)
        return out,reg_out
    
class MSA_yolov_visual(nn.Module):

    def __init__(self, dim,out_dim, num_heads=4, qkv_bias=False, attn_drop=0.,scale=25):
        super().__init__()
        self.msa = Attention_msa_visual(dim,num_heads,qkv_bias,attn_drop,scale=scale)
        self.linear1 = nn.Linear(2 * dim,2 * dim)
        self.linear2 =  nn.Linear(4 * dim,out_dim)

    def ave_pooling_over_ref(self,features,sort_results):
        key_feature = features[0]
        support_feature = features[0]
        if not self.training:
            sort_results = sort_results.to(features.dtype)
        soft_sim_feature = (sort_results@support_feature)#.transpose(1, 2)#torch.sum(softmax_value * most_sim_feature, dim=1)
        cls_feature = torch.cat([soft_sim_feature,key_feature],dim=-1)
        return cls_feature

    def forward(self,x_cls, x_reg, cls_score = None, fg_score = None, img = None, pred = None):
        trans_cls, trans_reg, sim_round2 = self.msa(x_cls,x_reg,cls_score,fg_score,img,pred)
        msa = self.linear1(trans_cls)
        ave = self.ave_pooling_over_ref(msa,sim_round2)
        out = self.linear2(ave)
        return out


class MSA_yolov_online(nn.Module):

    def __init__(self, dim,out_dim, num_heads=4, qkv_bias=False, attn_drop=0.,scale=25):
        super().__init__()
        self.msa = Attention_msa_online(dim,num_heads,qkv_bias,attn_drop,scale=scale)
        self.linear1 = nn.Linear(2 * dim,2 * dim)
        self.linear2 =  nn.Linear(4 * dim,out_dim)

    def ave_pooling_over_ref(self,features,sort_results):
        key_feature = features[0]
        support_feature = features[0]
        if not self.training:
            sort_results = sort_results.to(features.dtype)
        soft_sim_feature = (sort_results@support_feature)#.transpose(1, 2)#torch.sum(softmax_value * most_sim_feature, dim=1)
        cls_feature = torch.cat([soft_sim_feature,key_feature],dim=-1)

        return cls_feature

    def compute_geo_sim(self,key_preds,ref_preds):
        key_boxes = key_preds[:,:4]
        ref_boxes = ref_preds[:,:4]
        cost_giou, iou = generalized_box_iou(key_boxes.to(torch.float32), ref_boxes.to(torch.float32))

        return iou.to(torch.float16)

    def local_agg(self,features,local_results,boxes,cls_score,fg_score):
        local_features = local_results['msa']
        local_features_n = local_features / torch.norm(local_features, dim=-1, keepdim=True)
        features_n = features /torch.norm(features, dim=-1, keepdim=True)
        cos_sim = features_n@local_features_n.transpose(0,1)

        geo_sim = self.compute_geo_sim(boxes,local_results['boxes'])
        N = local_results['cls_scores'].shape[0]
        M = cls_score.shape[0]
        pre_scores = cls_score*fg_score
        pre_scores = torch.reshape(pre_scores, [-1, 1]).repeat(1, N)
        other_scores = local_results['cls_scores']*local_results['reg_scores']
        other_scores = torch.reshape(other_scores, [1, -1]).repeat(M, 1)
        ones_matrix = torch.ones([M,N]).to('cuda')
        zero_matrix = torch.zeros([M,N]).to('cuda')
        thresh_map = torch.where(other_scores-pre_scores>-0.3,ones_matrix,zero_matrix)
        local_sim = torch.softmax(25*cos_sim*thresh_map,dim=-1)*geo_sim
        local_sim = local_sim / torch.sum(local_sim, dim=-1, keepdim=True)
        local_sim = local_sim.to(features.dtype)
        sim_features = local_sim @ local_features

        return (sim_features+features)/2

    def forward(self,x_cls, x_reg, cls_score = None, fg_score = None,other_result = {},boxes=None, simN=30):
        trans_cls, trans_reg, sim_round2 = self.msa(x_cls,x_reg,cls_score,fg_score)
        msa = self.linear1(trans_cls)
        # if other_result != []:
        #     other_msa = other_result['msa'].unsqueeze(0)
        #     msa = torch.cat([msa,other_msa],dim=1)
        ave = self.ave_pooling_over_ref(msa,sim_round2)
        out = self.linear2(ave)
        if other_result != [] and other_result['local_results'] != []:
            lout = self.local_agg(out[:simN],other_result['local_results'],boxes[:simN],cls_score[:simN],fg_score[:simN])
            return lout,out
        return out,out













