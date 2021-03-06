'''
    Group + Dynamic + transformer  + Denosing
'''

import sys

sys.path.append('../')

# from model_common import common
# from torch import nn
# from model_common.transformer_module import VisionEncoder
'''
    The modules is form ipt, the self-attention module is from Pytorch Framework.

'''
import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import copy
import model_common.common as common


def make_model(args):
    return GDTD(args), 1


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class ResBlock(nn.Module):
    def __init__(
                self, conv, n_feats, kernel_size,
                bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


# can choose the type of conv, common or dynamic
class S_ResBlock(nn.Module):
    def __init__(
                self, conv, n_feats, kernel_size,
                bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(S_ResBlock, self).__init__()

        assert len(conv) == 2

        m = []

        for i in range(2):
            m.append(conv[i](n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class GDTD(nn.Module):

    def __init__(self, args, conv=default_conv):
        super(GDTD, self).__init__()

        self.scale_idx = 0

        self.args = args

        dyconv = common.dynamic_conv
        n_feats = args.n_feats
        kernel_size = 3
        act = nn.ReLU(True)

        # self.sub_mean = common.MeanShift(args.rgb_range)  # sub = ???
        # self.add_mean = common.MeanShift(args.rgb_range, sign=1)  # add = ???
        if self.args.flag == 0:
            self.head = nn.Sequential(
                    conv(args.n_colors, n_feats, kernel_size),  # conv1
                    ResBlock(conv, n_feats, 5, act=act),  # conv2
                    ResBlock(conv, n_feats, 5, act=act),  # conv3
                )

            self.body = VisionEncoder(img_dim=args.patch_size,
                                          patch_dim=args.patch_dim,
                                          num_channels=n_feats,
                                          embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                          num_heads=args.num_heads,
                                          num_layers=args.num_layers,
                                          hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                          dropout_rate=args.dropout_rate,
                                          mlp=args.no_mlp,
                                          pos_every=args.pos_every,
                                          no_pos=args.no_pos,
                                          no_norm=args.no_norm,
                                          no_residual=args.no_residual
                                          )
            self.tail = conv(n_feats, args.n_colors, kernel_size)

        if self.args.flag == 1:
            self.head1 = conv(args.n_colors, n_feats, kernel_size)
            self.head1_1 = nn.Sequential(
                conv(n_feats, n_feats, kernel_size),   # conv1
                conv(n_feats, n_feats, kernel_size)  # conv2
            )

            self.body1_1 = nn.Sequential(
                conv(n_feats, n_feats, kernel_size),   # conv1
                conv(n_feats, n_feats, kernel_size)  # conv2
            )

            self.fusion1_1 = nn.Sequential(
                conv(n_feats*2, n_feats*4, 1),  # conv1
                act,
                conv(n_feats * 4, n_feats, 1),  # conv1
            )

            self.body1_2 = conv(n_feats, n_feats, kernel_size)  # conv2

            self.body1_3 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=True,
                                         pos_every=args.pos_every,
                                         no_pos=args.no_pos,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            self.body2_1 = nn.Sequential(
                conv(n_feats, n_feats, kernel_size),   # conv1
                conv(n_feats, n_feats, kernel_size)  # conv2
            )

            self.fusion2_1 = nn.Sequential(
                conv(n_feats*2, n_feats*4, 1),  # conv1
                act,
                conv(n_feats * 4, n_feats, 1),  # conv1
            )

            self.body2_2 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=True,
                                         pos_every=args.pos_every,
                                         no_pos=args.no_pos,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            self.fusion2_2 = nn.Sequential(
                conv(n_feats*2, n_feats*4, 1),  # conv1
                act,
                conv(n_feats * 4, n_feats, 1),  # conv1
            )

            self.body2_3 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=True,
                                         pos_every=args.pos_every,
                                         no_pos=args.no_pos,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            self.body3_1 = nn.Sequential(
                conv(n_feats, n_feats, kernel_size),   # conv1
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=1,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              )
            )

            self.body3_2 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=2,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=True,
                                         pos_every=args.pos_every,
                                         no_pos=args.no_pos,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            self.fusion3_1 = nn.Sequential(
                conv(n_feats*3, n_feats*4, 1),  # conv1
                act,
                conv(n_feats * 4, n_feats, 1),  # conv1
            )

            self.body3_3 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=args.no_mlp,
                                         pos_every=args.pos_every,
                                         no_pos=True,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            self.tail = conv(n_feats, args.n_colors, kernel_size)

        if self.args.flag == 2:
            # ????????????+comm_conv
            self.head1 = conv(args.n_colors, n_feats, kernel_size)

            self.head1_1 = ResBlock(conv, n_feats, kernel_size, act=act)

            self.head1_3 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=True,
                                         pos_every=args.pos_every,
                                         no_pos=args.no_pos,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            self.body1_1 = nn.Sequential(
                ResBlock(conv, n_feats, kernel_size, act=act),
                conv(n_feats, n_feats, kernel_size),   # conv3
                act
            )

            self.body1_2 = nn.Sequential(
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=1,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              ),
                ResBlock(conv, n_feats, kernel_size, act=act)
            )

            self.body2_1 = nn.Sequential(
                ResBlock(conv, n_feats, kernel_size, act=act),
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=1,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              )
            )

            self.fusion2_1 = nn.Sequential(
                conv(n_feats*2, n_feats*4, 1),  # conv1
                act,
                conv(n_feats * 4, n_feats, 1),  # conv1
            )

            self.body2_2 = nn.Sequential(
                ResBlock(conv, n_feats, kernel_size, act=act),
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=1,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              )
            )

            self.fusion2_2 = nn.Sequential(
                conv(n_feats*2, n_feats*4, 1),  # conv1
                act,
                conv(n_feats * 4, n_feats, 1),  # conv1
            )

            self.body2_3 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=True,
                                         pos_every=args.pos_every,
                                         no_pos=args.no_pos,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            self.body3_1 = nn.Sequential(
                conv(n_feats, n_feats, kernel_size),   # conv1
                act,
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=2,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              )
            )

            self.fusion3_1 = nn.Sequential(
                conv(n_feats*3, n_feats*4, 1),  # conv1
                act,
                conv(n_feats * 4, n_feats, 1),  # conv1
            )

            self.body3_2 = nn.Sequential(
                conv(n_feats, n_feats, kernel_size),   # conv1
                act,
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=2,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              )
            )

            self.fusion3_2 = nn.Sequential(
                conv(n_feats*3, n_feats*4, 1),  # conv1
                act,
                conv(n_feats * 4, n_feats, 1),  # conv1
            )

            self.body3_3 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=args.no_mlp,
                                         pos_every=args.pos_every,
                                         no_pos=True,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            self.fusion3_3 = nn.Sequential(
                conv(n_feats*3, n_feats*4, 1),  # conv1
                act,
                conv(n_feats * 4, n_feats, 1),  # conv1
            )

            self.body3_4 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=args.no_mlp,
                                         pos_every=args.pos_every,
                                         no_pos=True,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            self.tail = conv(n_feats, args.n_colors, kernel_size)

        if self.args.flag == 3:
            self.head1 = conv(args.n_colors, n_feats, kernel_size)

            self.head1_1 = ResBlock(conv, n_feats, kernel_size, act=act)

            self.head1_3 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=True,
                                         pos_every=args.pos_every,
                                         no_pos=args.no_pos,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            self.body1_1 = nn.Sequential(
                ResBlock(conv, n_feats, kernel_size, act=act),
                conv(n_feats, n_feats, kernel_size),  # conv3
                act
            )

            self.body1_2 = nn.Sequential(
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=1,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              ),
                ResBlock(conv, n_feats, kernel_size, act=act)
            )

            self.body2_1 = nn.Sequential(
                ResBlock(conv, n_feats, kernel_size, act=act),
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=1,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              )
            )

            self.fusion2_1 = nn.Sequential(
                conv(n_feats * 2, n_feats * 4, 1),  # conv1
                act,
                conv(n_feats * 4, n_feats, 1),  # conv1
            )

            self.body2_2 = nn.Sequential(
                ResBlock(conv, n_feats, kernel_size, act=act),
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=1,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              )
            )

            self.fusion2_2 = nn.Sequential(
                conv(n_feats * 2, n_feats * 4, 1),  # conv1
                act,
                conv(n_feats * 4, n_feats, 1),  # conv1
            )

            self.body2_3 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=True,
                                         pos_every=args.pos_every,
                                         no_pos=args.no_pos,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            self.body3_1 = nn.Sequential(
                conv(n_feats, n_feats, kernel_size),  # conv1
                act,
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=2,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              )
            )

            self.fusion3_1 = nn.Sequential(
                conv(n_feats * 3, n_feats * 4, 1),  # conv1
                act,
                conv(n_feats * 4, n_feats, 1),  # conv1
            )

            self.body3_2 = nn.Sequential(
                conv(n_feats, n_feats, kernel_size),  # conv1
                act,
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=2,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              )
            )

            self.fusion3_2 = nn.Sequential(
                conv(n_feats * 3, n_feats * 4, 1),  # conv1
                act,
                conv(n_feats * 4, n_feats, 1),  # conv1
            )

            self.body3_3 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=args.no_mlp,
                                         pos_every=args.pos_every,
                                         no_pos=True,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            self.fusion3_3 = nn.Sequential(
                conv(n_feats * 3, n_feats * 4, 1),  # conv1
                act,
                conv(n_feats * 4, n_feats, 1),  # conv1
            )

            self.body3_4 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=args.no_mlp,
                                         pos_every=args.pos_every,
                                         no_pos=True,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            self.tail = conv(n_feats, args.n_colors, kernel_size)

        if self.args.flag == 4:
            self.head = nn.Sequential(
                conv(args.n_colors, n_feats, kernel_size),  # conv1
                ResBlock(conv, n_feats, 5, act=act),  # conv2
                ResBlock(conv, n_feats, 5, act=act),  # conv3
            )

            self.stage1_1_1 = VisionEncoder(img_dim=args.patch_size,
                                             patch_dim=args.patch_dim,
                                             num_channels=n_feats,
                                             embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                             num_heads=args.num_heads,
                                             num_layers=1,
                                             hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                             dropout_rate=args.dropout_rate,
                                             mlp=True,
                                             pos_every=args.pos_every,
                                             no_pos=args.no_pos,
                                             no_norm=args.no_norm,
                                             no_residual=args.no_residual
                                             )
            self.stage1_1_2 = nn.Sequential(conv(n_feats, n_feats, 3), act)

            self.stage1_2_1 = VisionEncoder(img_dim=args.patch_size,
                                            patch_dim=args.patch_dim,
                                            num_channels=n_feats,
                                            embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                            num_heads=args.num_heads,
                                            num_layers=1,
                                            hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                            dropout_rate=args.dropout_rate,
                                            mlp=True,
                                            pos_every=args.pos_every,
                                            no_pos=args.no_pos,
                                            no_norm=args.no_norm,
                                            no_residual=args.no_residual
                                            )
            self.stage1_2_2 = nn.Sequential(conv(n_feats, n_feats, 5), act)

            self.stage1_3_1 = VisionEncoder(img_dim=args.patch_size,
                                            patch_dim=args.patch_dim,
                                            num_channels=n_feats,
                                            embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                            num_heads=args.num_heads,
                                            num_layers=1,
                                            hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                            dropout_rate=args.dropout_rate,
                                            mlp=True,
                                            pos_every=args.pos_every,
                                            no_pos=args.no_pos,
                                            no_norm=args.no_norm,
                                            no_residual=args.no_residual
                                            )
            self.stage1_3_2 = nn.Sequential(conv(n_feats, n_feats, 3), act)

            self.stage1_4_1 = VisionEncoder(img_dim=args.patch_size,
                                            patch_dim=args.patch_dim,
                                            num_channels=n_feats,
                                            embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                            num_heads=args.num_heads,
                                            num_layers=1,
                                            hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                            dropout_rate=args.dropout_rate,
                                            mlp=True,
                                            pos_every=args.pos_every,
                                            no_pos=args.no_pos,
                                            no_norm=args.no_norm,
                                            no_residual=args.no_residual
                                            )
            self.stage1_4_2 = nn.Sequential(conv(n_feats, n_feats, 5), act)

            self.fusions123_2_s21 = self.fusion2_1 = nn.Sequential(
                conv(n_feats * 3, n_feats, 1),  # conv1
                act,
            )

            self.fusions234_2_s22 = self.fusion2_1 = nn.Sequential(
                conv(n_feats * 3, n_feats, 1),  # conv1
                act,
            )

            self.stage2_1_1 = VisionEncoder(img_dim=args.patch_size,
                                            patch_dim=args.patch_dim,
                                            num_channels=n_feats,
                                            embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                            num_heads=args.num_heads,
                                            num_layers=2,
                                            hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                            dropout_rate=args.dropout_rate,
                                            mlp=True,
                                            pos_every=args.pos_every,
                                            no_pos=args.no_pos,
                                            no_norm=args.no_norm,
                                            no_residual=args.no_residual
                                            )
            self.stage2_1_2 = nn.Sequential(conv(n_feats, n_feats, 5), act)

            self.stage2_2_1 = VisionEncoder(img_dim=args.patch_size,
                                            patch_dim=args.patch_dim,
                                            num_channels=n_feats,
                                            embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                            num_heads=args.num_heads,
                                            num_layers=2,
                                            hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                            dropout_rate=args.dropout_rate,
                                            mlp=True,
                                            pos_every=args.pos_every,
                                            no_pos=args.no_pos,
                                            no_norm=args.no_norm,
                                            no_residual=args.no_residual
                                            )
            self.stage2_2_2 = nn.Sequential(conv(n_feats, n_feats, 3), act)

            self.fusions2_2_s3 = self.fusion2_1 = nn.Sequential(
                conv(n_feats * 2, n_feats, 1),  # conv1
                act,
            )

            self.stage3_1 = VisionEncoder(img_dim=args.patch_size,
                                          patch_dim=args.patch_dim,
                                          num_channels=n_feats,
                                          embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                          num_heads=args.num_heads,
                                          num_layers=3,
                                          hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                          dropout_rate=args.dropout_rate,
                                          mlp=True,
                                          pos_every=args.pos_every,
                                          no_pos=args.no_pos,
                                          no_norm=args.no_norm,
                                          no_residual=args.no_residual
                                          )

            self.stage3_2 = nn.Sequential(conv(n_feats, n_feats, 3), act)

            self.tail = conv(n_feats, args.n_colors, kernel_size)

        if self.args.flag == 5:
            # ???????????? + conv
            self.head1 = conv(args.n_colors, n_feats, kernel_size)

            self.head1_1 = ResBlock(conv, n_feats, kernel_size, act=act)

            self.head1_3 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=True,
                                         pos_every=args.pos_every,
                                         no_pos=args.no_pos,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            self.body1_1 = nn.Sequential(
                ResBlock(conv, n_feats, kernel_size, act=act),
                conv(n_feats, n_feats, kernel_size),  # conv3
                act
            )

            self.body1_2 = nn.Sequential(
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=1,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              ),
                ResBlock(conv, n_feats, kernel_size, act=act)
            )

            self.body2_1 = nn.Sequential(
                ResBlock(conv, n_feats, kernel_size, act=act),
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=1,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              )
            )

            self.fusion1 = nn.Sequential(
                conv(n_feats, n_feats*4, 1),  # conv1
                act,
                conv(n_feats*4, n_feats, 1),  # conv1
            )

            self.body2_2 = nn.Sequential(
                ResBlock(conv, n_feats, kernel_size, act=act),
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=1,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              )
            )

            self.fusion2 = nn.Sequential(
                conv(n_feats, n_feats * 4, 1),  # conv1
                act,
                conv(n_feats * 4, n_feats, 1),  # conv1
            )

            self.body2_3 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=True,
                                         pos_every=args.pos_every,
                                         no_pos=args.no_pos,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            self.body3_1 = nn.Sequential(
                conv(n_feats, n_feats, kernel_size),  # conv1
                act,
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=2,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              )
            )

            self.fusion3 = nn.Sequential(
                conv(n_feats, n_feats * 4, 1),  # conv1
                act,
                conv(n_feats * 4, n_feats, 1),  # conv1
            )

            self.body3_2 = nn.Sequential(
                conv(n_feats, n_feats, kernel_size),  # conv1
                act,
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=2,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              )
            )

            self.fusion4 = nn.Sequential(
                conv(n_feats, n_feats * 4, 1),  # conv1
                act,
                conv(n_feats * 4, n_feats, 1),  # conv1
            )

            self.body3_3 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=args.no_mlp,
                                         pos_every=args.pos_every,
                                         no_pos=True,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            self.fusion5 = nn.Sequential(
                conv(n_feats, n_feats * 4, 1),  # conv1
                act,
                conv(n_feats * 4, n_feats, 1),  # conv1
            )

            self.body3_4 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=args.no_mlp,
                                         pos_every=args.pos_every,
                                         no_pos=True,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            self.tail = conv(n_feats, args.n_colors, kernel_size)

        if self.args.flag == 6:
            # ???????????? + dyconv
            self.head1 = conv(args.n_colors, n_feats, kernel_size)

            self.head1_1 = ResBlock(conv, n_feats, kernel_size, act=act)

            self.head1_3 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=True,
                                         pos_every=args.pos_every,
                                         no_pos=args.no_pos,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            self.body1_1 = nn.Sequential(
                ResBlock(conv, n_feats, kernel_size, act=act),
                dyconv(n_feats, n_feats, kernel_size),   # conv3
                act
            )

            self.body1_2 = nn.Sequential(
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=1,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              ),
                S_ResBlock([dyconv, conv], n_feats, kernel_size, act=act)
            )

            self.body2_1 = nn.Sequential(
                S_ResBlock([conv, dyconv], n_feats, kernel_size, act=act),
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=1,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              )
            )

            self.fusion2_1 = nn.Sequential(
                conv(n_feats*2, n_feats*4, 1),  # conv1
                act,
                conv(n_feats * 4, n_feats, 1),  # conv1
            )

            self.body2_2 = nn.Sequential(
                S_ResBlock([dyconv, conv], n_feats, kernel_size, act=act),
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=1,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              )
            )

            self.fusion2_2 = nn.Sequential(
                conv(n_feats*2, n_feats*4, 1),  # conv1
                act,
                conv(n_feats * 4, n_feats, 1),  # conv1
            )

            self.body2_3 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=True,
                                         pos_every=args.pos_every,
                                         no_pos=args.no_pos,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            self.body3_1 = nn.Sequential(
                dyconv(n_feats, n_feats, kernel_size),   # conv1
                act,
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=2,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              )
            )

            self.fusion3_1 = nn.Sequential(
                conv(n_feats*3, n_feats*4, 1),  # conv1
                act,
                conv(n_feats * 4, n_feats, 1),  # conv1
            )

            self.body3_2 = nn.Sequential(
                dyconv(n_feats, n_feats, kernel_size),   # conv1
                act,
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=2,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              )
            )

            self.fusion3_2 = nn.Sequential(
                conv(n_feats*3, n_feats*4, 1),  # conv1
                act,
                conv(n_feats * 4, n_feats, 1),  # conv1
            )

            self.body3_3 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=args.no_mlp,
                                         pos_every=args.pos_every,
                                         no_pos=True,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            self.fusion3_3 = nn.Sequential(
                conv(n_feats*3, n_feats*4, 1),  # conv1
                act,
                conv(n_feats * 4, n_feats, 1),  # conv1
            )

            self.body3_4 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=args.no_mlp,
                                         pos_every=args.pos_every,
                                         no_pos=True,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            self.tail = conv(n_feats, args.n_colors, kernel_size)

        if self.args.flag == 7:
            # ????????????+comm_conv + ??????fusion
            self.head1 = conv(args.n_colors, n_feats, kernel_size)

            self.head1_1 = ResBlock(conv, n_feats, kernel_size, act=act)

            self.head1_3 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=True,
                                         pos_every=args.pos_every,
                                         no_pos=args.no_pos,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            self.body1_1 = nn.Sequential(
                ResBlock(conv, n_feats, kernel_size, act=act),
                conv(n_feats, n_feats, kernel_size),   # conv3
                act
            )

            self.body1_2 = nn.Sequential(
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=1,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              ),
                ResBlock(conv, n_feats, kernel_size, act=act)
            )

            self.body2_1 = nn.Sequential(
                ResBlock(conv, n_feats, kernel_size, act=act),
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=1,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              )
            )

            self.fusion2_1 = nn.Sequential(
                # DConv
                nn.Conv2d(n_feats*2, n_feats*2, kernel_size=5, padding=2, groups=n_feats*2),
                conv(n_feats * 2, n_feats, 1)  # conv1
            )

            self.body2_2 = nn.Sequential(
                ResBlock(conv, n_feats, kernel_size, act=act),
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=1,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              )
            )

            self.fusion2_2 = nn.Sequential(
                nn.Conv2d(n_feats*2, n_feats*2, kernel_size=5, padding=2, groups=n_feats*2),
                conv(n_feats * 2, n_feats, 1),  # conv1
            )

            self.body2_3 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=True,
                                         pos_every=args.pos_every,
                                         no_pos=args.no_pos,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            self.body3_1 = nn.Sequential(
                conv(n_feats, n_feats, kernel_size),   # conv1
                act,
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=2,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              )
            )

            self.fusion3_1 = nn.Sequential(
                nn.Conv2d(n_feats*2, n_feats*2, kernel_size=5, padding=2, groups=n_feats*2),
                conv(n_feats * 2, n_feats, 1),  # conv1
            )

            self.body3_2 = nn.Sequential(
                conv(n_feats, n_feats, kernel_size),   # conv1
                act,
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=2,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              )
            )

            self.fusion3_2 = nn.Sequential(
                nn.Conv2d(n_feats*2, n_feats*2, kernel_size=5, padding=2, groups=n_feats*2),
                conv(n_feats * 2, n_feats, 1),  # conv1
            )

            self.body3_3 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=args.no_mlp,
                                         pos_every=args.pos_every,
                                         no_pos=True,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            self.fusion3_3 = nn.Sequential(
                nn.Conv2d(n_feats*3, n_feats*3, kernel_size=5, padding=2, groups=n_feats*3),
                conv(n_feats * 3, n_feats, 1),  # conv1
            )

            self.body3_4 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=args.no_mlp,
                                         pos_every=args.pos_every,
                                         no_pos=True,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            self.tail = conv(n_feats, args.n_colors, kernel_size)

        if self.args.flag == 8:
            # ????????????+comm_conv + ??????fusion
            self.head1 = conv(args.n_colors, n_feats, kernel_size)

            self.head1_1 = ResBlock(conv, n_feats, kernel_size, act=act)

            self.head1_3 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=True,
                                         pos_every=args.pos_every,
                                         no_pos=args.no_pos,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            self.body1_1 = nn.Sequential(
                ResBlock(conv, n_feats, kernel_size, act=act),
                conv(n_feats, n_feats, kernel_size),   # conv3
                act
            )

            self.body1_2 = nn.Sequential(
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=1,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              ),
                ResBlock(conv, n_feats, kernel_size, act=act)
            )

            self.body2_1 = nn.Sequential(
                ResBlock(conv, n_feats, kernel_size, act=act),
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=1,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              )
            )

            self.fusion1 = nn.Sequential(
                # DConv
                nn.Conv2d(n_feats*3, n_feats*3, kernel_size=5, padding=2, groups=n_feats*3),
                conv(n_feats * 3, n_feats, 1)  # conv1
            )

            self.body2_2 = nn.Sequential(
                ResBlock(conv, n_feats, kernel_size, act=act),
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=1,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              )
            )

            self.fusion2 = nn.Sequential(
                nn.Conv2d(n_feats*3, n_feats*3, kernel_size=5, padding=2, groups=n_feats*3),
                conv(n_feats * 3, n_feats, 1),  # conv1
            )

            self.body2_3 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=True,
                                         pos_every=args.pos_every,
                                         no_pos=args.no_pos,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            self.body3_1 = nn.Sequential(
                conv(n_feats, n_feats, kernel_size),   # conv1
                act,
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=2,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              )
            )

            self.fusion3 = nn.Sequential(
                nn.Conv2d(n_feats*3, n_feats*3, kernel_size=5, padding=2, groups=n_feats*3),
                act,
                conv(n_feats * 3, n_feats, 1),  # conv1
            )

            self.body3_2 = nn.Sequential(
                conv(n_feats, n_feats, kernel_size),   # conv1
                act,
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=2,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              )
            )

            self.body3_3 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=args.no_mlp,
                                         pos_every=args.pos_every,
                                         no_pos=True,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            self.body3_4 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=args.no_mlp,
                                         pos_every=args.pos_every,
                                         no_pos=True,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            self.tail = conv(n_feats, args.n_colors, kernel_size)

        if self.args.flag == 9:
            # ????????????+comm_conv + ??????fusion + ??????fusion
            self.head1 = conv(args.n_colors, n_feats, kernel_size)

            self.head1_1 = ResBlock(conv, n_feats, kernel_size, act=act)

            self.head1_3 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=True,
                                         pos_every=args.pos_every,
                                         no_pos=args.no_pos,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            self.body1_1 = nn.Sequential(
                ResBlock(conv, n_feats, kernel_size, act=act),
                conv(n_feats, n_feats, kernel_size),   # conv3
                act
            )

            self.fusion1_1 = nn.Sequential(
                # DConv
                nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=5, padding=2, groups=n_feats * 2),
                conv(n_feats * 2, n_feats, 1)  # conv1
            )

            self.body1_2 = nn.Sequential(
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=1,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              ),
                ResBlock(conv, n_feats, kernel_size, act=act)
            )

            self.fusion1_2 = nn.Sequential(
                # DConv
                nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=5, padding=2, groups=n_feats * 2),
                conv(n_feats * 2, n_feats, 1)  # conv1
            )

            self.body2_1 = nn.Sequential(
                ResBlock(conv, n_feats, kernel_size, act=act),
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=1,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              )
            )

            self.fusion2_1_1 = nn.Sequential(
                # DConv
                nn.Conv2d(n_feats*2, n_feats*2, kernel_size=5, padding=2, groups=n_feats*2),
                conv(n_feats * 2, n_feats, 1)  # conv1
            )

            self.fusion2_1_2 = nn.Sequential(
                # DConv
                nn.Conv2d(n_feats*2, n_feats*2, kernel_size=5, padding=2, groups=n_feats*2),
                conv(n_feats * 2, n_feats, 1)  # conv1
            )

            self.body2_2 = nn.Sequential(
                ResBlock(conv, n_feats, kernel_size, act=act),
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=1,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              )
            )

            self.fusion2_2_1 = nn.Sequential(
                nn.Conv2d(n_feats*2, n_feats*2, kernel_size=5, padding=2, groups=n_feats*2),
                conv(n_feats * 2, n_feats, 1),  # conv1
            )

            self.fusion2_2_2 = nn.Sequential(
                nn.Conv2d(n_feats*2, n_feats*2, kernel_size=5, padding=2, groups=n_feats*2),
                conv(n_feats * 2, n_feats, 1),  # conv1
            )

            self.body2_3 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=True,
                                         pos_every=args.pos_every,
                                         no_pos=args.no_pos,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            self.body3_1 = nn.Sequential(
                conv(n_feats, n_feats, kernel_size),   # conv1
                act,
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=2,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              )
            )

            self.fusion3_1_1 = nn.Sequential(
                nn.Conv2d(n_feats*2, n_feats*2, kernel_size=5, padding=2, groups=n_feats*2),
                conv(n_feats * 2, n_feats, 1),  # conv1
            )

            self.fusion3_1_2 = nn.Sequential(
                nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=5, padding=2, groups=n_feats * 2),
                conv(n_feats * 2, n_feats, 1),  # conv1
            )

            self.body3_2 = nn.Sequential(
                conv(n_feats, n_feats, kernel_size),   # conv1
                act,
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=2,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              )
            )

            self.fusion3_2_1 = nn.Sequential(
                nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=5, padding=2, groups=n_feats * 2),
                conv(n_feats * 2, n_feats, 1),  # conv1
            )

            self.fusion3_2_2 = nn.Sequential(
                nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=5, padding=2, groups=n_feats * 2),
                conv(n_feats * 2, n_feats, 1),  # conv1
            )

            self.body3_3 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=args.no_mlp,
                                         pos_every=args.pos_every,
                                         no_pos=True,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            self.fusion3_3 = nn.Sequential(
                nn.Conv2d(n_feats * 3, n_feats * 3, kernel_size=5, padding=2, groups=n_feats * 3),
                conv(n_feats * 3, n_feats, 1),  # conv1
            )

            self.body3_4 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=args.no_mlp,
                                         pos_every=args.pos_every,
                                         no_pos=True,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            self.tail = conv(n_feats, args.n_colors, kernel_size)

        if self.args.flag == 10:
            # ????????????+comm_conv + ??????fusion + ??????fusion
            self.head1 = conv(args.n_colors, n_feats, kernel_size)

            self.head1_1 = ResBlock(conv, n_feats, kernel_size, act=act)

            self.head1_3 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=True,
                                         pos_every=args.pos_every,
                                         no_pos=args.no_pos,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            self.body1_1 = nn.Sequential(
                ResBlock(conv, n_feats, kernel_size, act=act),
                conv(n_feats, n_feats, kernel_size),   # conv3
                act
            )

            self.body1_2 = nn.Sequential(
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=1,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              ),
                ResBlock(conv, n_feats, kernel_size, act=act)
            )

            self.body2_1 = nn.Sequential(
                ResBlock(conv, n_feats, kernel_size, act=act),
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=1,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              )
            )

            self.fusion1 = nn.Sequential(
                # DConv
                nn.Conv2d(n_feats*3, n_feats*3, kernel_size=5, padding=2, groups=n_feats*3),
                conv(n_feats * 3, n_feats, 1)  # conv1
            )

            self.body2_2 = nn.Sequential(
                ResBlock(conv, n_feats, kernel_size, act=act),
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=1,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              )
            )

            self.fusion2 = nn.Sequential(
                nn.Conv2d(n_feats*3, n_feats*3, kernel_size=5, padding=2, groups=n_feats*3),
                conv(n_feats * 3, n_feats, 1),  # conv1
            )

            self.body2_3 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=True,
                                         pos_every=args.pos_every,
                                         no_pos=args.no_pos,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            self.body3_1 = nn.Sequential(
                conv(n_feats, n_feats, kernel_size),   # conv1
                act,
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=2,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              )
            )

            self.fusion3 = nn.Sequential(
                nn.Conv2d(n_feats*3, n_feats*3, kernel_size=5, padding=2, groups=n_feats*3),
                act,
                conv(n_feats * 3, n_feats, 1),  # conv1
            )

            self.body3_2 = nn.Sequential(
                conv(n_feats, n_feats, kernel_size),   # conv1
                act,
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=2,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              )
            )

            self.body3_3 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=args.no_mlp,
                                         pos_every=args.pos_every,
                                         no_pos=True,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            self.body3_4 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=args.no_mlp,
                                         pos_every=args.pos_every,
                                         no_pos=True,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            self.tail = conv(n_feats, args.n_colors, kernel_size)

        if self.args.flag == 11:
            # ????????????+comm_conv + ??????fusion
            self.head1 = conv(args.n_colors, n_feats, kernel_size)

            self.head1_1 = ResBlock(conv, n_feats, kernel_size, act=act)

            self.head1_3 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=True,
                                         pos_every=args.pos_every,
                                         no_pos=args.no_pos,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            self.body1_1 = nn.Sequential(
                ResBlock(conv, n_feats, kernel_size, act=act),
                conv(n_feats, n_feats, kernel_size),  # conv3
                act
            )

            self.body1_2 = nn.Sequential(
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=1,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              ),
                ResBlock(conv, n_feats, kernel_size, act=act)
            )

            self.body2_1 = nn.Sequential(
                ResBlock(conv, n_feats, kernel_size, act=act),
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=1,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              )
            )

            # self.fusion2_1 = nn.Sequential(
            #     # DConv
            #     nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=5, padding=2, groups=n_feats * 2),
            #     conv(n_feats * 2, n_feats, 1)  # conv1
            # )

            self.body2_2 = nn.Sequential(
                ResBlock(conv, n_feats, kernel_size, act=act),
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=1,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              )
            )

            self.fusion2_2 = nn.Sequential(
                nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=5, padding=2, groups=n_feats * 2),
                conv(n_feats * 2, n_feats, 1),  # conv1
            )

            self.body2_3 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=True,
                                         pos_every=args.pos_every,
                                         no_pos=args.no_pos,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            self.body3_1 = nn.Sequential(
                conv(n_feats, n_feats, kernel_size),  # conv1
                act,
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=2,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              )
            )

            # self.fusion3_1 = nn.Sequential(
            #     nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=5, padding=2, groups=n_feats * 2),
            #     act,
            #     conv(n_feats * 2, n_feats, 1),  # conv1
            # )

            self.body3_2 = nn.Sequential(
                conv(n_feats, n_feats, kernel_size),  # conv1
                act,
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=2,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              )
            )

            self.fusion3_2 = nn.Sequential(
                nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=5, padding=2, groups=n_feats * 2),
                conv(n_feats * 2, n_feats, 1),  # conv1
            )

            self.body3_3 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=args.no_mlp,
                                         pos_every=args.pos_every,
                                         no_pos=True,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            self.fusion3_3 = nn.Sequential(
                nn.Conv2d(n_feats * 3, n_feats * 3, kernel_size=5, padding=2, groups=n_feats * 3),
                conv(n_feats * 3, n_feats, 1),  # conv1
            )

            self.body3_4 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=args.no_mlp,
                                         pos_every=args.pos_every,
                                         no_pos=True,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            self.tail = conv(n_feats, args.n_colors, kernel_size)

        if self.args.flag == 12:
            # ????????????+comm_conv + ??????fusion
            self.head1 = conv(args.n_colors, n_feats, kernel_size)

            self.head1_1 = ResBlock(conv, n_feats, kernel_size, act=act)

            self.head1_3 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=True,
                                         pos_every=args.pos_every,
                                         no_pos=args.no_pos,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            self.body1_1 = nn.Sequential(
                ResBlock(conv, n_feats, kernel_size, act=act),
                conv(n_feats, n_feats, kernel_size),  # conv3
                act
            )

            self.body1_2 = nn.Sequential(
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=1,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              ),
                ResBlock(conv, n_feats, kernel_size, act=act)
            )

            self.body2_1 = nn.Sequential(
                ResBlock(conv, n_feats, kernel_size, act=act),
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=1,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              )
            )

            # self.fusion2_1 = nn.Sequential(
            #     # DConv
            #     nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=5, padding=2, groups=n_feats * 2),
            #     conv(n_feats * 2, n_feats, 1)  # conv1
            # )

            self.body2_2 = nn.Sequential(
                ResBlock(conv, n_feats, kernel_size, act=act),
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=1,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              )
            )

            # self.fusion2_2 = nn.Sequential(
            #     nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=5, padding=2, groups=n_feats * 2),
            #     conv(n_feats * 2, n_feats, 1),  # conv1
            # )

            self.body2_3 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=True,
                                         pos_every=args.pos_every,
                                         no_pos=args.no_pos,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            self.body3_1 = nn.Sequential(
                conv(n_feats, n_feats, kernel_size),  # conv1
                act,
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=2,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              )
            )

            # self.fusion3_1 = nn.Sequential(
            #     nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=5, padding=2, groups=n_feats * 2),
            #     act,
            #     conv(n_feats * 2, n_feats, 1),  # conv1
            # )

            self.body3_2 = nn.Sequential(
                conv(n_feats, n_feats, kernel_size),  # conv1
                act,
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=2,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              )
            )

            # self.fusion3_2 = nn.Sequential(
            #     nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=5, padding=2, groups=n_feats * 2),
            #     conv(n_feats * 2, n_feats, 1),  # conv1
            # )

            self.body3_3 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=args.no_mlp,
                                         pos_every=args.pos_every,
                                         no_pos=True,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            self.fusion3_3 = nn.Sequential(
                nn.Conv2d(n_feats * 3, n_feats * 3, kernel_size=5, padding=2, groups=n_feats * 3),
                conv(n_feats * 3, n_feats, 1),  # conv1
            )

            self.body3_4 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=args.no_mlp,
                                         pos_every=args.pos_every,
                                         no_pos=True,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            self.tail = conv(n_feats, args.n_colors, kernel_size)

        if self.args.flag == 13:
            # ????????????+comm_conv + ??????fusion
            self.head1 = conv(args.n_colors, n_feats, kernel_size)

            self.head1_1 = ResBlock(conv, n_feats, kernel_size, act=act)

            self.head1_3 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=True,
                                         pos_every=args.pos_every,
                                         no_pos=args.no_pos,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            # self.body1_1 = nn.Sequential(
            #     ResBlock(conv, n_feats, kernel_size, act=act),
            #     conv(n_feats, n_feats, kernel_size),  # conv3
            #     act
            # )
            #
            # self.body1_2 = nn.Sequential(
            #     VisionEncoder(img_dim=args.patch_size,
            #                   patch_dim=args.patch_dim,
            #                   num_channels=n_feats,
            #                   embedding_dim=n_feats * args.patch_dim * args.patch_dim,
            #                   num_heads=args.num_heads,
            #                   num_layers=1,
            #                   hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
            #                   dropout_rate=args.dropout_rate,
            #                   mlp=True,
            #                   pos_every=args.pos_every,
            #                   no_pos=args.no_pos,
            #                   no_norm=args.no_norm,
            #                   no_residual=args.no_residual
            #                   ),
            #     ResBlock(conv, n_feats, kernel_size, act=act)
            # )

            self.body2_1 = nn.Sequential(
                ResBlock(conv, n_feats, kernel_size, act=act),
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=1,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              )
            )

            # self.fusion2_1 = nn.Sequential(
            #     # DConv
            #     nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=5, padding=2, groups=n_feats * 2),
            #     conv(n_feats * 2, n_feats, 1)  # conv1
            # )

            self.body2_2 = nn.Sequential(
                ResBlock(conv, n_feats, kernel_size, act=act),
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=1,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              )
            )

            # self.fusion2_2 = nn.Sequential(
            #     nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=5, padding=2, groups=n_feats * 2),
            #     conv(n_feats * 2, n_feats, 1),  # conv1
            # )

            self.body2_3 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=True,
                                         pos_every=args.pos_every,
                                         no_pos=args.no_pos,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            self.body3_1 = nn.Sequential(
                conv(n_feats, n_feats, kernel_size),  # conv1
                act,
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=2,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              )
            )

            # self.fusion3_1 = nn.Sequential(
            #     nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=5, padding=2, groups=n_feats * 2),
            #     act,
            #     conv(n_feats * 2, n_feats, 1),  # conv1
            # )

            self.body3_2 = nn.Sequential(
                conv(n_feats, n_feats, kernel_size),  # conv1
                act,
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=2,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              )
            )

            # self.fusion3_2 = nn.Sequential(
            #     nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=5, padding=2, groups=n_feats * 2),
            #     conv(n_feats * 2, n_feats, 1),  # conv1
            # )

            self.body3_3 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=args.no_mlp,
                                         pos_every=args.pos_every,
                                         no_pos=True,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            self.fusion3_3 = nn.Sequential(
                nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=5, padding=2, groups=n_feats * 2),
                conv(n_feats * 2, n_feats, 1),  # conv1
            )

            self.body3_4 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=args.no_mlp,
                                         pos_every=args.pos_every,
                                         no_pos=True,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            self.tail = conv(n_feats, args.n_colors, kernel_size)

        if self.args.flag == 14:
            # ????????????+comm_conv + ??????fusion
            self.head1 = conv(args.n_colors, n_feats, kernel_size)

            self.head1_1 = ResBlock(conv, n_feats, kernel_size, act=act)

            self.head1_3 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=True,
                                         pos_every=args.pos_every,
                                         no_pos=args.no_pos,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            # self.body1_1 = nn.Sequential(
            #     ResBlock(conv, n_feats, kernel_size, act=act),
            #     conv(n_feats, n_feats, kernel_size),  # conv3
            #     act
            # )
            #
            # self.body1_2 = nn.Sequential(
            #     VisionEncoder(img_dim=args.patch_size,
            #                   patch_dim=args.patch_dim,
            #                   num_channels=n_feats,
            #                   embedding_dim=n_feats * args.patch_dim * args.patch_dim,
            #                   num_heads=args.num_heads,
            #                   num_layers=1,
            #                   hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
            #                   dropout_rate=args.dropout_rate,
            #                   mlp=True,
            #                   pos_every=args.pos_every,
            #                   no_pos=args.no_pos,
            #                   no_norm=args.no_norm,
            #                   no_residual=args.no_residual
            #                   ),
            #     ResBlock(conv, n_feats, kernel_size, act=act)
            # )

            # self.body2_1 = nn.Sequential(
            #     ResBlock(conv, n_feats, kernel_size, act=act),
            #     VisionEncoder(img_dim=args.patch_size,
            #                   patch_dim=args.patch_dim,
            #                   num_channels=n_feats,
            #                   embedding_dim=n_feats * args.patch_dim * args.patch_dim,
            #                   num_heads=args.num_heads,
            #                   num_layers=1,
            #                   hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
            #                   dropout_rate=args.dropout_rate,
            #                   mlp=True,
            #                   pos_every=args.pos_every,
            #                   no_pos=args.no_pos,
            #                   no_norm=args.no_norm,
            #                   no_residual=args.no_residual
            #                   )
            # )

            # self.fusion2_1 = nn.Sequential(
            #     # DConv
            #     nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=5, padding=2, groups=n_feats * 2),
            #     conv(n_feats * 2, n_feats, 1)  # conv1
            # )

            # self.body2_2 = nn.Sequential(
            #     ResBlock(conv, n_feats, kernel_size, act=act),
            #     VisionEncoder(img_dim=args.patch_size,
            #                   patch_dim=args.patch_dim,
            #                   num_channels=n_feats,
            #                   embedding_dim=n_feats * args.patch_dim * args.patch_dim,
            #                   num_heads=args.num_heads,
            #                   num_layers=1,
            #                   hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
            #                   dropout_rate=args.dropout_rate,
            #                   mlp=True,
            #                   pos_every=args.pos_every,
            #                   no_pos=args.no_pos,
            #                   no_norm=args.no_norm,
            #                   no_residual=args.no_residual
            #                   )
            # )

            # self.fusion2_2 = nn.Sequential(
            #     nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=5, padding=2, groups=n_feats * 2),
            #     conv(n_feats * 2, n_feats, 1),  # conv1
            # )

            # self.body2_3 = VisionEncoder(img_dim=args.patch_size,
            #                              patch_dim=args.patch_dim,
            #                              num_channels=n_feats,
            #                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
            #                              num_heads=args.num_heads,
            #                              num_layers=1,
            #                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
            #                              dropout_rate=args.dropout_rate,
            #                              mlp=True,
            #                              pos_every=args.pos_every,
            #                              no_pos=args.no_pos,
            #                              no_norm=args.no_norm,
            #                              no_residual=args.no_residual
            #                              )

            self.body3_1 = nn.Sequential(
                conv(n_feats, n_feats, kernel_size),  # conv1
                act,
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=2,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              )
            )

            # self.fusion3_1 = nn.Sequential(
            #     nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=5, padding=2, groups=n_feats * 2),
            #     act,
            #     conv(n_feats * 2, n_feats, 1),  # conv1
            # )

            self.body3_2 = nn.Sequential(
                conv(n_feats, n_feats, kernel_size),  # conv1
                act,
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=2,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              )
            )

            # self.fusion3_2 = nn.Sequential(
            #     nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=5, padding=2, groups=n_feats * 2),
            #     conv(n_feats * 2, n_feats, 1),  # conv1
            # )

            self.body3_3 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=args.no_mlp,
                                         pos_every=args.pos_every,
                                         no_pos=True,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            self.fusion3_3 = nn.Sequential(
                nn.Conv2d(n_feats, n_feats, kernel_size=5, padding=2, groups=n_feats),
                conv(n_feats, n_feats, 1),  # conv1
            )

            self.body3_4 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=args.no_mlp,
                                         pos_every=args.pos_every,
                                         no_pos=True,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            self.tail = conv(n_feats, args.n_colors, kernel_size)

        if self.args.flag == 15:
            # ????????????+comm_conv + ??????fusion
            self.head1 = conv(args.n_colors, n_feats, kernel_size)

            self.head1_1 = ResBlock(conv, n_feats, kernel_size, act=act)

            self.head1_3 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=True,
                                         pos_every=args.pos_every,
                                         no_pos=args.no_pos,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            # self.body1_1 = nn.Sequential(
            #     ResBlock(conv, n_feats, kernel_size, act=act),
            #     conv(n_feats, n_feats, kernel_size),  # conv3
            #     act
            # )
            #
            # self.body1_2 = nn.Sequential(
            #     VisionEncoder(img_dim=args.patch_size,
            #                   patch_dim=args.patch_dim,
            #                   num_channels=n_feats,
            #                   embedding_dim=n_feats * args.patch_dim * args.patch_dim,
            #                   num_heads=args.num_heads,
            #                   num_layers=1,
            #                   hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
            #                   dropout_rate=args.dropout_rate,
            #                   mlp=True,
            #                   pos_every=args.pos_every,
            #                   no_pos=args.no_pos,
            #                   no_norm=args.no_norm,
            #                   no_residual=args.no_residual
            #                   ),
            #     ResBlock(conv, n_feats, kernel_size, act=act)
            # )

            # self.body2_1 = nn.Sequential(
            #     ResBlock(conv, n_feats, kernel_size, act=act),
            #     VisionEncoder(img_dim=args.patch_size,
            #                   patch_dim=args.patch_dim,
            #                   num_channels=n_feats,
            #                   embedding_dim=n_feats * args.patch_dim * args.patch_dim,
            #                   num_heads=args.num_heads,
            #                   num_layers=1,
            #                   hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
            #                   dropout_rate=args.dropout_rate,
            #                   mlp=True,
            #                   pos_every=args.pos_every,
            #                   no_pos=args.no_pos,
            #                   no_norm=args.no_norm,
            #                   no_residual=args.no_residual
            #                   )
            # )

            # self.fusion2_1 = nn.Sequential(
            #     # DConv
            #     nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=5, padding=2, groups=n_feats * 2),
            #     conv(n_feats * 2, n_feats, 1)  # conv1
            # )

            # self.body2_2 = nn.Sequential(
            #     ResBlock(conv, n_feats, kernel_size, act=act),
            #     VisionEncoder(img_dim=args.patch_size,
            #                   patch_dim=args.patch_dim,
            #                   num_channels=n_feats,
            #                   embedding_dim=n_feats * args.patch_dim * args.patch_dim,
            #                   num_heads=args.num_heads,
            #                   num_layers=1,
            #                   hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
            #                   dropout_rate=args.dropout_rate,
            #                   mlp=True,
            #                   pos_every=args.pos_every,
            #                   no_pos=args.no_pos,
            #                   no_norm=args.no_norm,
            #                   no_residual=args.no_residual
            #                   )
            # )

            # self.fusion2_2 = nn.Sequential(
            #     nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=5, padding=2, groups=n_feats * 2),
            #     conv(n_feats * 2, n_feats, 1),  # conv1
            # )

            # self.body2_3 = VisionEncoder(img_dim=args.patch_size,
            #                              patch_dim=args.patch_dim,
            #                              num_channels=n_feats,
            #                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
            #                              num_heads=args.num_heads,
            #                              num_layers=1,
            #                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
            #                              dropout_rate=args.dropout_rate,
            #                              mlp=True,
            #                              pos_every=args.pos_every,
            #                              no_pos=args.no_pos,
            #                              no_norm=args.no_norm,
            #                              no_residual=args.no_residual
            #                              )

            self.body3_1 = nn.Sequential(
                conv(n_feats, n_feats, kernel_size),  # conv1
                act,
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=2,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              )
            )

            # self.fusion3_1 = nn.Sequential(
            #     nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=5, padding=2, groups=n_feats * 2),
            #     act,
            #     conv(n_feats * 2, n_feats, 1),  # conv1
            # )

            self.body3_2 = nn.Sequential(
                conv(n_feats, n_feats, kernel_size),  # conv1
                act,
                VisionEncoder(img_dim=args.patch_size,
                              patch_dim=args.patch_dim,
                              num_channels=n_feats,
                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                              num_heads=args.num_heads,
                              num_layers=2,
                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                              dropout_rate=args.dropout_rate,
                              mlp=True,
                              pos_every=args.pos_every,
                              no_pos=args.no_pos,
                              no_norm=args.no_norm,
                              no_residual=args.no_residual
                              )
            )

            # self.fusion3_2 = nn.Sequential(
            #     nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=5, padding=2, groups=n_feats * 2),
            #     conv(n_feats * 2, n_feats, 1),  # conv1
            # )

            self.body3_3 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=args.no_mlp,
                                         pos_every=args.pos_every,
                                         no_pos=True,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            self.fusion3_3 = nn.Sequential(
                nn.Conv2d(n_feats, n_feats, kernel_size=5, padding=2, groups=n_feats),
                conv(n_feats, n_feats, 1),  # conv1
            )

            self.body3_4 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                         num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=args.no_mlp,
                                         pos_every=args.pos_every,
                                         no_pos=True,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            self.tail = conv(n_feats, args.n_colors, kernel_size)

        if self.args.flag == 16:
            # ????????????+comm_conv + ??????fusion
            self.head1 = conv(args.n_colors, n_feats, kernel_size)

            self.head1_1 = ResBlock(conv, n_feats, kernel_size, act=act)

            # self.head1_3 = VisionEncoder(img_dim=args.patch_size,
            #                              patch_dim=args.patch_dim,
            #                              num_channels=n_feats,
            #                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
            #                                        num_heads=args.num_heads,
            #                              num_layers=1,
            #                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
            #                              dropout_rate=args.dropout_rate,
            #                              mlp=True,
            #                              pos_every=args.pos_every,
            #                              no_pos=args.no_pos,
            #                              no_norm=args.no_norm,
            #                              no_residual=args.no_residual
            #                              )

            # self.body1_1 = nn.Sequential(
            #     ResBlock(conv, n_feats, kernel_size, act=act),
            #     conv(n_feats, n_feats, kernel_size),  # conv3
            #     act
            # )
            #
            # self.body1_2 = nn.Sequential(
            #     VisionEncoder(img_dim=args.patch_size,
            #                   patch_dim=args.patch_dim,
            #                   num_channels=n_feats,
            #                   embedding_dim=n_feats * args.patch_dim * args.patch_dim,
            #                   num_heads=args.num_heads,
            #                   num_layers=1,
            #                   hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
            #                   dropout_rate=args.dropout_rate,
            #                   mlp=True,
            #                   pos_every=args.pos_every,
            #                   no_pos=args.no_pos,
            #                   no_norm=args.no_norm,
            #                   no_residual=args.no_residual
            #                   ),
            #     ResBlock(conv, n_feats, kernel_size, act=act)
            # )

            # self.body2_1 = nn.Sequential(
            #     ResBlock(conv, n_feats, kernel_size, act=act),
            #     VisionEncoder(img_dim=args.patch_size,
            #                   patch_dim=args.patch_dim,
            #                   num_channels=n_feats,
            #                   embedding_dim=n_feats * args.patch_dim * args.patch_dim,
            #                   num_heads=args.num_heads,
            #                   num_layers=1,
            #                   hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
            #                   dropout_rate=args.dropout_rate,
            #                   mlp=True,
            #                   pos_every=args.pos_every,
            #                   no_pos=args.no_pos,
            #                   no_norm=args.no_norm,
            #                   no_residual=args.no_residual
            #                   )
            # )

            # self.fusion2_1 = nn.Sequential(
            #     # DConv
            #     nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=5, padding=2, groups=n_feats * 2),
            #     conv(n_feats * 2, n_feats, 1)  # conv1
            # )

            # self.body2_2 = nn.Sequential(
            #     ResBlock(conv, n_feats, kernel_size, act=act),
            #     VisionEncoder(img_dim=args.patch_size,
            #                   patch_dim=args.patch_dim,
            #                   num_channels=n_feats,
            #                   embedding_dim=n_feats * args.patch_dim * args.patch_dim,
            #                   num_heads=args.num_heads,
            #                   num_layers=1,
            #                   hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
            #                   dropout_rate=args.dropout_rate,
            #                   mlp=True,
            #                   pos_every=args.pos_every,
            #                   no_pos=args.no_pos,
            #                   no_norm=args.no_norm,
            #                   no_residual=args.no_residual
            #                   )
            # )

            # self.fusion2_2 = nn.Sequential(
            #     nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=5, padding=2, groups=n_feats * 2),
            #     conv(n_feats * 2, n_feats, 1),  # conv1
            # )

            # self.body2_3 = VisionEncoder(img_dim=args.patch_size,
            #                              patch_dim=args.patch_dim,
            #                              num_channels=n_feats,
            #                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
            #                              num_heads=args.num_heads,
            #                              num_layers=1,
            #                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
            #                              dropout_rate=args.dropout_rate,
            #                              mlp=True,
            #                              pos_every=args.pos_every,
            #                              no_pos=args.no_pos,
            #                              no_norm=args.no_norm,
            #                              no_residual=args.no_residual
            #                              )

            self.body3_1 = nn.Sequential(
                conv(n_feats, n_feats, kernel_size),  # conv1
                act
            )

            # self.fusion3_1 = nn.Sequential(
            #     nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=5, padding=2, groups=n_feats * 2),
            #     act,
            #     conv(n_feats * 2, n_feats, 1),  # conv1
            # )

            self.body3_2 = nn.Sequential(
                conv(n_feats, n_feats, kernel_size),  # conv1
                act
            )

            # self.fusion3_2 = nn.Sequential(
            #     nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=5, padding=2, groups=n_feats * 2),
            #     conv(n_feats * 2, n_feats, 1),  # conv1
            # )

            # self.body3_3 = VisionEncoder(img_dim=args.patch_size,
            #                              patch_dim=args.patch_dim,
            #                              num_channels=n_feats,
            #                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
            #                              num_heads=args.num_heads,
            #                              num_layers=1,
            #                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
            #                              dropout_rate=args.dropout_rate,
            #                              mlp=args.no_mlp,
            #                              pos_every=args.pos_every,
            #                              no_pos=True,
            #                              no_norm=args.no_norm,
            #                              no_residual=args.no_residual
            #                              )
            #
            # self.fusion3_3 = nn.Sequential(
            #     nn.Conv2d(n_feats, n_feats, kernel_size=5, padding=2, groups=n_feats),
            #     conv(n_feats, n_feats, 1),  # conv1
            # )

            # self.body3_4 = VisionEncoder(img_dim=args.patch_size,
            #                              patch_dim=args.patch_dim,
            #                              num_channels=n_feats,
            #                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
            #                              num_heads=args.num_heads,
            #                              num_layers=1,
            #                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
            #                              dropout_rate=args.dropout_rate,
            #                              mlp=args.no_mlp,
            #                              pos_every=args.pos_every,
            #                              no_pos=True,
            #                              no_norm=args.no_norm,
            #                              no_residual=args.no_residual
            #                              )

            self.tail = conv(n_feats, args.n_colors, kernel_size)

        if self.args.flag == 17:
            # ????????????+comm_conv + ??????fusion
            self.head1 = conv(args.n_colors, n_feats, kernel_size)

            self.head1_1 = ResBlock(conv, n_feats, kernel_size, act=act)

            self.head1_3 = VisionEncoder(img_dim=args.patch_size,
                                         patch_dim=args.patch_dim,
                                         num_channels=n_feats,
                                         embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                                   num_heads=args.num_heads,
                                         num_layers=1,
                                         hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                         dropout_rate=args.dropout_rate,
                                         mlp=True,
                                         pos_every=args.pos_every,
                                         no_pos=args.no_pos,
                                         no_norm=args.no_norm,
                                         no_residual=args.no_residual
                                         )

            # self.body1_1 = nn.Sequential(
            #     ResBlock(conv, n_feats, kernel_size, act=act),
            #     conv(n_feats, n_feats, kernel_size),  # conv3
            #     act
            # )
            #
            # self.body1_2 = nn.Sequential(
            #     VisionEncoder(img_dim=args.patch_size,
            #                   patch_dim=args.patch_dim,
            #                   num_channels=n_feats,
            #                   embedding_dim=n_feats * args.patch_dim * args.patch_dim,
            #                   num_heads=args.num_heads,
            #                   num_layers=1,
            #                   hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
            #                   dropout_rate=args.dropout_rate,
            #                   mlp=True,
            #                   pos_every=args.pos_every,
            #                   no_pos=args.no_pos,
            #                   no_norm=args.no_norm,
            #                   no_residual=args.no_residual
            #                   ),
            #     ResBlock(conv, n_feats, kernel_size, act=act)
            # )

            # self.body2_1 = nn.Sequential(
            #     ResBlock(conv, n_feats, kernel_size, act=act),
            #     VisionEncoder(img_dim=args.patch_size,
            #                   patch_dim=args.patch_dim,
            #                   num_channels=n_feats,
            #                   embedding_dim=n_feats * args.patch_dim * args.patch_dim,
            #                   num_heads=args.num_heads,
            #                   num_layers=1,
            #                   hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
            #                   dropout_rate=args.dropout_rate,
            #                   mlp=True,
            #                   pos_every=args.pos_every,
            #                   no_pos=args.no_pos,
            #                   no_norm=args.no_norm,
            #                   no_residual=args.no_residual
            #                   )
            # )

            # self.fusion2_1 = nn.Sequential(
            #     # DConv
            #     nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=5, padding=2, groups=n_feats * 2),
            #     conv(n_feats * 2, n_feats, 1)  # conv1
            # )

            # self.body2_2 = nn.Sequential(
            #     ResBlock(conv, n_feats, kernel_size, act=act),
            #     VisionEncoder(img_dim=args.patch_size,
            #                   patch_dim=args.patch_dim,
            #                   num_channels=n_feats,
            #                   embedding_dim=n_feats * args.patch_dim * args.patch_dim,
            #                   num_heads=args.num_heads,
            #                   num_layers=1,
            #                   hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
            #                   dropout_rate=args.dropout_rate,
            #                   mlp=True,
            #                   pos_every=args.pos_every,
            #                   no_pos=args.no_pos,
            #                   no_norm=args.no_norm,
            #                   no_residual=args.no_residual
            #                   )
            # )

            # self.fusion2_2 = nn.Sequential(
            #     nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=5, padding=2, groups=n_feats * 2),
            #     conv(n_feats * 2, n_feats, 1),  # conv1
            # )

            # self.body2_3 = VisionEncoder(img_dim=args.patch_size,
            #                              patch_dim=args.patch_dim,
            #                              num_channels=n_feats,
            #                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
            #                              num_heads=args.num_heads,
            #                              num_layers=1,
            #                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
            #                              dropout_rate=args.dropout_rate,
            #                              mlp=True,
            #                              pos_every=args.pos_every,
            #                              no_pos=args.no_pos,
            #                              no_norm=args.no_norm,
            #                              no_residual=args.no_residual
            #                              )

            self.body3_1 = nn.Sequential(
                conv(n_feats, n_feats, kernel_size),  # conv1
                act
            )

            # self.fusion3_1 = nn.Sequential(
            #     nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=5, padding=2, groups=n_feats * 2),
            #     act,
            #     conv(n_feats * 2, n_feats, 1),  # conv1
            # )

            self.body3_2 = nn.Sequential(
                conv(n_feats, n_feats, kernel_size),  # conv1
                act
            )

            # self.fusion3_2 = nn.Sequential(
            #     nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=5, padding=2, groups=n_feats * 2),
            #     conv(n_feats * 2, n_feats, 1),  # conv1
            # )

            # self.body3_3 = VisionEncoder(img_dim=args.patch_size,
            #                              patch_dim=args.patch_dim,
            #                              num_channels=n_feats,
            #                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
            #                              num_heads=args.num_heads,
            #                              num_layers=1,
            #                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
            #                              dropout_rate=args.dropout_rate,
            #                              mlp=args.no_mlp,
            #                              pos_every=args.pos_every,
            #                              no_pos=True,
            #                              no_norm=args.no_norm,
            #                              no_residual=args.no_residual
            #                              )
            #
            # self.fusion3_3 = nn.Sequential(
            #     nn.Conv2d(n_feats, n_feats, kernel_size=5, padding=2, groups=n_feats),
            #     conv(n_feats, n_feats, 1),  # conv1
            # )

            # self.body3_4 = VisionEncoder(img_dim=args.patch_size,
            #                              patch_dim=args.patch_dim,
            #                              num_channels=n_feats,
            #                              embedding_dim=n_feats * args.patch_dim * args.patch_dim,
            #                              num_heads=args.num_heads,
            #                              num_layers=1,
            #                              hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
            #                              dropout_rate=args.dropout_rate,
            #                              mlp=args.no_mlp,
            #                              pos_every=args.pos_every,
            #                              no_pos=True,
            #                              no_norm=args.no_norm,
            #                              no_residual=args.no_residual
            #                              )

            self.tail = conv(n_feats, args.n_colors, kernel_size)

    def forward(self, x):
        y = x

        if self.args.flag == 0:
            pass
        elif self.args.flag == 1:
            x = self.head1(x)
            x = x + self.head1_1(x)

            group1 = self.body1_1(x)
            group2 = self.body2_1(x)
            group3 = self.body3_1(x)

            group1 = self.body1_2(self.fusion1_1(torch.cat((group1, group3), 1)))
            group2 = self.body2_2(self.fusion2_1(torch.cat((group2, group3), 1)))
            group3 = self.body3_2(group3)

            group2 = self.body2_3(self.fusion2_2(torch.cat((group1, group2), 1)))
            group1 = self.body1_3(group1)
            group3 = self.body3_3(self.fusion3_1(torch.cat((group1, group2, group3), 1)))

            x = group3

        elif self.args.flag == 2:
            x = self.head1(x)
            x = self.head1_1(x)
            x = self.head1_3(x)

            group1 = self.body1_1(x)
            group2 = self.body2_1(x)
            group3 = self.body3_1(x)

            group3 = self.body3_2(self.fusion3_1(torch.cat((group1, group2, group3), 1)))
            group2 = self.body2_2(self.fusion2_1(torch.cat((group1, group2), 1)))
            group1 = self.body1_2(group1)

            group3 = self.body3_3(self.fusion3_2(torch.cat((group1, group2, group3), 1)))
            group2 = self.body2_3(self.fusion2_2(torch.cat((group1, group2), 1)))

            group3 = self.body3_4(self.fusion3_3(torch.cat((group1, group2, group3), 1)))

            x = group3

        elif self.args.flag == 3:
            x = self.head1(x)
            x = x + self.head1_1(x)
            x = self.head1_3(x)

            group1 = self.body1_1(x)
            group2 = self.body2_1(x)
            group3 = self.body3_1(x)

            group3 = self.body3_2(self.fusion3_1(torch.cat((group1, group2, group3), 1)))
            group2 = self.body2_2(self.fusion2_1(torch.cat((group1, group2), 1)))
            group1 = self.body1_2(group1)

            group3 = self.body3_3(self.fusion3_2(torch.cat((group1, group2, group3), 1)))
            group2 = self.body2_3(self.fusion2_2(torch.cat((group1, group2), 1)))

            group3 = self.body3_4(self.fusion3_3(torch.cat((group1, group2, group3), 1)))

            x = group3

        elif self.args.flag == 4:
            x = self.head(x)

            x1 = self.stage1_1_2(x + self.stage1_1_1(x))
            x2 = self.stage1_2_2(x + self.stage1_2_1(x))
            x3 = self.stage1_3_2(x + self.stage1_3_1(x))
            x4 = self.stage1_4_2(x + self.stage1_4_1(x))

            x1 = self.fusions123_2_s21(torch.cat((x1, x2, x3), dim=1))
            x2 = self.fusions234_2_s22(torch.cat((x2, x3, x4), dim=1))

            x1 = self.stage2_1_2(x1 + self.stage2_1_1(x1))
            x2 = self.stage2_2_2(x2 + self.stage2_2_1(x2))

            x = self.fusions2_2_s3(torch.cat((x1, x2), dim=1))

            x = self.stage3_2(x + self.stage3_1(x))

        elif self.args.flag == 5:
            x = self.head1(x)
            x = self.head1_1(x)
            x = self.head1_3(x)

            x = self.body1_1(x)
            x = self.body1_2(x)
            x = self.body2_1(x)
            x = self.fusion1(x)
            x = self.body2_2(x)
            x = self.fusion2(x)
            x = self.body2_3(x)
            x = self.body3_1(x)
            x = self.fusion3(x)
            x = self.body3_2(x)
            x = self.fusion4(x)
            x = self.body3_3(x)
            x = self.fusion5(x)
            x = self.body3_4(x)

        elif self.args.flag == 6:
            x = self.head1(x)
            x = self.head1_1(x)
            x = self.head1_3(x)

            group1 = self.body1_1(x)
            group2 = self.body2_1(x)
            group3 = self.body3_1(x)

            group3 = self.body3_2(self.fusion3_1(torch.cat((group1, group2, group3), 1)))
            group2 = self.body2_2(self.fusion2_1(torch.cat((group1, group2), 1)))
            group1 = self.body1_2(group1)

            group3 = self.body3_3(self.fusion3_2(torch.cat((group1, group2, group3), 1)))
            group2 = self.body2_3(self.fusion2_2(torch.cat((group1, group2), 1)))

            group3 = self.body3_4(self.fusion3_3(torch.cat((group1, group2, group3), 1)))

            x = group3

        elif self.args.flag == 7:
            x = self.head1(x)
            x = self.head1_1(x)
            x = self.head1_3(x)

            group1 = self.body1_1(x)
            group2 = self.body2_1(x)
            group3 = self.body3_1(x)

            group2 = self.body2_2(self.fusion2_1(torch.cat((group1, group2), 1)))
            group3 = self.body3_2(self.fusion3_1(torch.cat((group2, group3), 1)))
            group1 = self.body1_2(group1)

            group2 = self.body2_3(self.fusion2_2(torch.cat((group1, group2), 1)))
            group3 = self.body3_3(self.fusion3_2(torch.cat((group2, group3), 1)))

            group3 = self.body3_4(self.fusion3_3(torch.cat((group1, group2, group3), 1)))

            x = group3

        elif self.args.flag == 8:
            x = self.head1(x)
            x = self.head1_1(x)
            x = self.head1_3(x)

            group1 = self.body1_1(x)
            group2 = self.body2_1(x)
            group3 = self.body3_1(x)

            x = self.fusion1(torch.cat((group1, group2, group3), 1))
            group2 = self.body2_2(x)
            group3 = self.body3_2(x)
            group1 = self.body1_2(x)

            x = self.fusion2(torch.cat((group1, group2, group3), 1))
            group2 = self.body2_3(x)
            group3 = self.body3_3(x)

            x = self.fusion3(torch.cat((group1, group2, group3), 1))
            group3 = self.body3_4(x)

            x = group3

        elif self.args.flag == 9:
            x = self.head1(x)
            x = self.head1_1(x)
            x = self.head1_3(x)

            stream1 = self.fusion1_1(torch.cat((self.body1_1(x), x), dim=1))
            stream2 = self.fusion2_1_1(torch.cat((self.body2_1(x), x), dim=1))
            stream3 = self.fusion3_1_1(torch.cat((self.body3_1(x), x), dim=1))

            fusion2_1 = self.fusion2_1_2(torch.cat((stream1, stream2), 1))
            stream2 = self.fusion2_2_1(torch.cat((self.body2_2(fusion2_1), fusion2_1), dim=1))
            fusion3_1 = self.fusion3_1_2(torch.cat((fusion2_1, stream3), 1))
            stream3 = self.fusion3_2_1(torch.cat((self.body3_2(fusion3_1), fusion3_1), dim=1))
            stream1 = self.fusion1_2(torch.cat((self.body1_2(stream1), stream1), dim=1))

            fusion2_2 = self.fusion2_2_2(torch.cat((stream1, stream2), dim=1))
            stream2 = self.body2_3(fusion2_2)
            fusion3_2 = self.fusion3_2_2(torch.cat((fusion2_2, stream3), dim=1))
            stream3 = self.body3_3(fusion3_2)

            stream3 = self.body3_4(self.fusion3_3(torch.cat((stream1, stream2, stream3), dim=1)))

            x = stream3

        elif self.args.flag == 10:
            x = self.head1(x)
            x = self.head1_1(x)
            x = self.head1_3(x)

            group1 = x + self.body1_1(x)
            group2 = x + self.body2_1(x)
            group3 = x + self.body3_1(x)

            x = self.fusion1(torch.cat((group1, group2, group3), 1))
            group2 = x + self.body2_2(x)
            group3 = x + self.body3_2(x)
            group1 = x + self.body1_2(x)

            x = self.fusion2(torch.cat((group1, group2, group3), 1))
            group2 = x + self.body2_3(x)
            group3 = x + self.body3_3(x)

            x = self.fusion3(torch.cat((group1, group2, group3), 1))
            group3 = self.body3_4(x)

            x = group3

        elif self.args.flag == 11:
            x = self.head1(x)
            x = self.head1_1(x)
            x = self.head1_3(x)

            group1 = self.body1_1(x)
            group2 = self.body2_1(x)
            group3 = self.body3_1(x)

            group2 = self.body2_2(group2)
            group3 = self.body3_2(group3)
            group1 = self.body1_2(group1)

            group2 = self.body2_3(self.fusion2_2(torch.cat((group1, group2), 1)))
            group3 = self.body3_3(self.fusion3_2(torch.cat((group2, group3), 1)))

            group3 = self.body3_4(self.fusion3_3(torch.cat((group1, group2, group3), 1)))

            x = group3

        elif self.args.flag == 12:
            x = self.head1(x)
            x = self.head1_1(x)
            x = self.head1_3(x)

            group1 = self.body1_1(x)
            group2 = self.body2_1(x)
            group3 = self.body3_1(x)

            group2 = self.body2_2(group2)
            group3 = self.body3_2(group3)
            group1 = self.body1_2(group1)

            group2 = self.body2_3(group2)
            group3 = self.body3_3(group3)

            group3 = self.body3_4(self.fusion3_3(torch.cat((group1, group2, group3), 1)))

            x = group3

        elif self.args.flag == 13:
            x = self.head1(x)
            x = self.head1_1(x)
            x = self.head1_3(x)

            # group1 = self.body1_1(x)
            group2 = self.body2_1(x)
            group3 = self.body3_1(x)

            group2 = self.body2_2(group2)
            group3 = self.body3_2(group3)
            # group1 = self.body1_2(group1)

            group2 = self.body2_3(group2)
            group3 = self.body3_3(group3)

            group3 = self.body3_4(self.fusion3_3(torch.cat((group2, group3), 1)))

            x = group3

        elif self.args.flag == 14:
            x = self.head1(x)
            x = self.head1_1(x)
            x = self.head1_3(x)

            # group1 = self.body1_1(x)
            # group2 = self.body2_1(x)
            group3 = self.body3_1(x)

            # group2 = self.body2_2(group2)
            group3 = self.body3_2(group3)
            # group1 = self.body1_2(group1)

            # group2 = self.body2_3(group2)
            group3 = self.body3_3(group3)

            group3 = self.body3_4(self.fusion3_3(group3))

            x = group3

        elif self.args.flag == 15:
            x = self.head1(x)
            x = self.head1_1(x)
            x = self.head1_3(x)

            # group1 = self.body1_1(x)
            # group2 = self.body2_1(x)
            group3 = self.body3_1(x)

            # group2 = self.body2_2(group2)
            group3 = self.body3_2(group3)
            # group1 = self.body1_2(group1)

            # group2 = self.body2_3(group2)
            group3 = self.body3_3(group3)

            group3 = self.body3_4(group3)

            x = group3

        elif self.args.flag == 16:
            x = self.head1(x)
            x = self.head1_1(x)

            # group1 = self.body1_1(x)
            # group2 = self.body2_1(x)
            group3 = self.body3_1(x)

            # group2 = self.body2_2(group2)
            group3 = self.body3_2(group3)
            # group1 = self.body1_2(group1)

            # group2 = self.body2_3(group2)
            # group3 = self.body3_3(group3)
            #
            # group3 = self.body3_4(group3)

            x = group3

        elif self.args.flag == 17:
            x = self.head1(x)
            x = self.head1_1(x)
            x = self.head1_3(x)

            # group1 = self.body1_1(x)
            # group2 = self.body2_1(x)
            # group3 = self.body3_1(x)

            # group2 = self.body2_2(group2)
            # group3 = self.body3_2(group3)
            # group1 = self.body1_2(group1)

            # group2 = self.body2_3(group2)
            # group3 = self.body3_3(group3)
            #
            # group3 = self.body3_4(group3)

            # x = group3
        out = self.tail(x)

        return y - out


class VisionTransformer(nn.Module):
    def __init__(
            self,
            img_dim,
            patch_dim,
            num_channels,
            embedding_dim,
            num_heads,
            num_layers,
            hidden_dim,
            num_queries,
            positional_encoding_type="learned",
            dropout_rate=0,
            no_norm=False,
            mlp=False,
            pos_every=False,
            no_pos=False,
            no_residual=False
    ):
        super(VisionTransformer, self).__init__()

        assert embedding_dim % num_heads == 0
        assert img_dim % patch_dim == 0
        self.no_norm = no_norm
        self.mlp = mlp
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.num_channels = num_channels

        self.img_dim = img_dim
        self.pos_every = pos_every
        self.num_patches = int((img_dim // patch_dim) ** 2)  # (W*H)/P^2
        self.seq_length = self.num_patches
        self.flatten_dim = patch_dim * patch_dim * num_channels  # C*P^2

        self.out_dim = patch_dim * patch_dim * num_channels

        self.no_pos = no_pos
        self.no_residual = no_residual

        if self.mlp == False:
            self.linear_encoding = nn.Linear(self.flatten_dim, embedding_dim)
            self.mlp_head = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.Dropout(dropout_rate),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.out_dim),
                nn.Dropout(dropout_rate)
            )

            self.query_embed = nn.Embedding(num_queries, embedding_dim * self.seq_length)
            # embedding_dim * self.seq_length ????

        encoder_layer = TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim, dropout_rate, self.no_norm)
        self.encoder = TransformerEncoder(encoder_layer, num_layers, self.no_residual)

        decoder_layer = TransformerDecoderLayer(embedding_dim, num_heads, hidden_dim, dropout_rate, self.no_norm)
        self.decoder = TransformerDecoder(decoder_layer, num_layers, self.no_residual)

        if not self.no_pos:
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )

        self.dropout_layer1 = nn.Dropout(dropout_rate)

        if no_norm:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=1 / m.weight.size(1))

    def forward(self, x, query_idx=0, con=False):

        x = torch.nn.functional.unfold(x, self.patch_dim, stride=self.patch_dim).transpose(1, 2).transpose(0, 1).contiguous()  # shape == (time, B, d_model)

        if self.mlp == False:
            x = self.dropout_layer1(self.linear_encoding(x)) + x

            query_embed = self.query_embed.weight[query_idx].view(-1, 1, self.embedding_dim).repeat(1, x.size(1), 1)
        else:
            query_embed = None

        if not self.no_pos:
            pos = self.position_encoding(x).transpose(0, 1)

        if self.pos_every:
            x = self.encoder(x, pos=pos)
            x = self.decoder(x, x, pos=pos, query_pos=query_embed)
        elif self.no_pos:
            x = self.encoder(x)
            x = self.decoder(x, x, query_pos=query_embed)
        else:
            x = self.encoder(x + pos)
            x = self.decoder(x, x, query_pos=query_embed)

        if self.mlp == False:
            x = self.mlp_head(x) + x

        x = x.transpose(0, 1).contiguous().view(x.size(1), -1, self.flatten_dim)

        if con:
            con_x = x
            x = torch.nn.functional.fold(x.transpose(1, 2).contiguous(), int(self.img_dim), self.patch_dim,
                                         stride=self.patch_dim)
            return x, con_x

        x = torch.nn.functional.fold(x.transpose(1, 2).contiguous(), int(self.img_dim), self.patch_dim,
                                     stride=self.patch_dim)

        return x


class VisionEncoder(nn.Module):
    def __init__(
            self,
            img_dim,
            patch_dim,
            num_channels,
            embedding_dim,
            num_heads,
            num_layers,
            hidden_dim,
            dropout_rate=0,
            no_norm=False,
            mlp=False,
            pos_every=False,
            no_pos=False,
            no_residual=False
    ):
        super(VisionEncoder, self).__init__()

        assert embedding_dim % num_heads == 0
        assert img_dim % patch_dim == 0
        self.no_norm = no_norm
        self.mlp = mlp
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.num_channels = num_channels

        self.img_dim = img_dim
        self.pos_every = pos_every
        self.num_patches = int((img_dim // patch_dim) ** 2)  # (W*H)/P^2
        self.seq_length = self.num_patches
        self.flatten_dim = patch_dim * patch_dim * num_channels  # C*P^2

        self.out_dim = patch_dim * patch_dim * num_channels

        self.no_pos = no_pos
        self.no_residual = no_residual

        if self.mlp == False: # ??????mlp???????????????encoder?????????
            self.linear_encoding = nn.Linear(self.flatten_dim, embedding_dim)
            self.mlp_head = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.Dropout(dropout_rate),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.out_dim),
                nn.Dropout(dropout_rate)
            )

        encoder_layer = TransformerEncoderLayer(patch_dim, embedding_dim, num_heads, hidden_dim, dropout_rate, self.no_norm)

        self.encoder = TransformerEncoder(encoder_layer, num_layers, self.no_residual)


        if not self.no_pos:
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )

        self.dropout_layer1 = nn.Dropout(dropout_rate)

        if no_norm:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=1 / m.weight.size(1))

    def forward(self, x, con=False, mask=None):

        x = torch.nn.functional.unfold(x, self.patch_dim, stride=self.patch_dim).transpose(1, 2).transpose(0, 1).contiguous()  # shape == (time, B, d_model)

        if self.mlp == False:
            x = self.dropout_layer1(self.linear_encoding(x)) + x
            # query_embed = self.query_embed.weight[query_idx].view(-1, 1, self.embedding_dim).repeat(1, x.size(1), 1)
        else:
            pass
            # query_embed = None

        if not self.no_pos:
            pos = self.position_encoding(x).transpose(0, 1)

        if self.pos_every:
            x = self.encoder(x, pos=pos, mask=mask)
            # x = self.decoder(x, x, pos=pos, query_pos=query_embed)
        elif self.no_pos:
            x = self.encoder(x, mask)
            # x = self.decoder(x, x, query_pos=query_embed)
        else:
            x = self.encoder(x + pos, mask)
            # x = self.decoder(x, x, query_pos=query_embed)

        if self.mlp == False:
            x = self.mlp_head(x) + x

        x = x.transpose(0, 1).contiguous().view(x.size(1), -1, self.flatten_dim)

        if con:
            con_x = x
            x = torch.nn.functional.fold(x.transpose(1, 2).contiguous(), int(self.img_dim), self.patch_dim,
                                         stride=self.patch_dim)
            return x, con_x

        x = torch.nn.functional.fold(x.transpose(1, 2).contiguous(), int(self.img_dim), self.patch_dim,
                                     stride=self.patch_dim)

        return x


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()
        self.pe = nn.Embedding(max_position_embeddings, embedding_dim)
        self.seq_length = seq_length

        self.register_buffer(
            "position_ids", torch.arange(self.seq_length).expand((1, -1))
        )

    def forward(self, x, position_ids=None):
        if position_ids is None:
            position_ids = self.position_ids[:, : self.seq_length]  # self.position_ids???????

        position_embeddings = self.pe(position_ids)
        return position_embeddings


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, no_residual=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.no_residual = no_residual

    def forward(self, src, pos=None, mask=None):
        output = src

        if self.no_residual or len(self.layers) < 4:
            for layer in self.layers:
                output = layer(output, pos=pos, mask=mask)
        else:  # encoder use residual struct
            layers = iter(self.layers)
            # ????????? ?????? ????????????
            output1 = next(layers)(output, pos=pos, mask=mask)
            output2 = next(layers)(output1, pos=pos, mask=mask)
            output3 = next(layers)(output2, pos=pos, mask=mask)
            output4 = next(layers)(output3, pos=pos, mask=mask)
            output = output + output1 + output2 + output3 + output4

            for layer in layers:
                output = layer(output, pos=pos, mask=mask)

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, patch_dim, d_model, nhead, dim_feedforward=2048, dropout=0.1, no_norm=False,
                 activation="relu"):
        super().__init__()
        self.patch_dim = patch_dim

        # multihead attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        nn.init.kaiming_uniform_(self.self_attn.in_proj_weight, a=math.sqrt(5))

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos=None, mask=None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)

        src2 = self.self_attn(q, k, src2)

        src = src + self.dropout1(src2[0])
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, no_residual=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory, pos=None, query_pos=None):
        output = tgt

        for layer in self.layers:
            output = layer(output, memory, pos=pos, query_pos=query_pos)

        return output


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, no_norm=False,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, pos=None, query_pos=None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2)[0]  # [0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

# '''
#     Group + Dynamic + transformer  + Denosing
# '''
#
# import sys
#
# sys.path.append('../')
#
# '''
#     The modules is form ipt, the self-attention module is from Pytorch Framework.
#
# '''
# import math
# import torch
# import torch.nn.functional as F
# from torch import nn, Tensor
# import copy
#
#
# def make_model(args):
#     return GDTD(args), 1
#
#
# def default_conv(in_channels, out_channels, kernel_size, bias=True):
#     return nn.Conv2d(
#         in_channels, out_channels, kernel_size,
#         padding=(kernel_size//2), bias=bias)
#
#
# class ResBlock(nn.Module):
#     def __init__(
#                 self, conv, n_feats, kernel_size,
#                 bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
#
#         super(ResBlock, self).__init__()
#         m = []
#         for i in range(2):
#             m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
#             if bn:
#                 m.append(nn.BatchNorm2d(n_feats))
#             if i == 0:
#                 m.append(act)
#
#         self.body = nn.Sequential(*m)
#         self.res_scale = res_scale
#
#     def forward(self, x):
#         res = self.body(x).mul(self.res_scale)
#         res += x
#
#         return res
#
#
# class GDTD(nn.Module):
#
#     def __init__(self, args, conv=default_conv):
#         super(GDTD, self).__init__()
#
#         n_feats = args.n_feats
#         kernel_size = 3
#         act = nn.ReLU(True)
#
#         self.head1 = conv(args.n_colors, n_feats, kernel_size)
#
#         self.head1_1 = ResBlock(conv, n_feats, kernel_size, act=act)
#
#         self.head1_3 = VisionEncoder(img_dim=args.patch_size,
#                                      patch_dim=args.patch_dim,
#                                      num_channels=n_feats,
#                                      embedding_dim=n_feats * args.patch_dim * args.patch_dim,
#                                      num_heads=args.num_heads,
#                                      num_layers=1,
#                                      hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
#                                      dropout_rate=args.dropout_rate,
#                                      mlp=True,
#                                      pos_every=args.pos_every,
#                                      no_pos=args.no_pos,
#                                      no_norm=args.no_norm,
#                                      no_residual=args.no_residual
#                                      )
#
#         self.body1_1 = nn.Sequential(
#             ResBlock(conv, n_feats, kernel_size, act=act),
#             conv(n_feats, n_feats, kernel_size),   # conv3
#             act
#         )
#
#         self.body1_2 = nn.Sequential(
#             VisionEncoder(img_dim=args.patch_size,
#                           patch_dim=args.patch_dim,
#                           num_channels=n_feats,
#                           embedding_dim=n_feats * args.patch_dim * args.patch_dim,
#                           num_heads=args.num_heads,
#                           num_layers=1,
#                           hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
#                           dropout_rate=args.dropout_rate,
#                           mlp=True,
#                           pos_every=args.pos_every,
#                           no_pos=args.no_pos,
#                           no_norm=args.no_norm,
#                           no_residual=args.no_residual
#                           ),
#             ResBlock(conv, n_feats, kernel_size, act=act)
#         )
#
#         self.body2_1 = nn.Sequential(
#             ResBlock(conv, n_feats, kernel_size, act=act),
#             VisionEncoder(img_dim=args.patch_size,
#                           patch_dim=args.patch_dim,
#                           num_channels=n_feats,
#                           embedding_dim=n_feats * args.patch_dim * args.patch_dim,
#                           num_heads=args.num_heads,
#                           num_layers=1,
#                           hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
#                           dropout_rate=args.dropout_rate,
#                           mlp=True,
#                           pos_every=args.pos_every,
#                           no_pos=args.no_pos,
#                           no_norm=args.no_norm,
#                           no_residual=args.no_residual
#                           )
#         )
#
#         self.fusion2_1 = nn.Sequential(
#             conv(n_feats*2, n_feats*4, 1),  # conv1
#             act,
#             conv(n_feats * 4, n_feats, 1),  # conv1
#         )
#
#         self.body2_2 = nn.Sequential(
#             ResBlock(conv, n_feats, kernel_size, act=act),
#             VisionEncoder(img_dim=args.patch_size,
#                           patch_dim=args.patch_dim,
#                           num_channels=n_feats,
#                           embedding_dim=n_feats * args.patch_dim * args.patch_dim,
#                           num_heads=args.num_heads,
#                           num_layers=1,
#                           hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
#                           dropout_rate=args.dropout_rate,
#                           mlp=True,
#                           pos_every=args.pos_every,
#                           no_pos=args.no_pos,
#                           no_norm=args.no_norm,
#                           no_residual=args.no_residual
#                           )
#         )
#
#         self.fusion2_2 = nn.Sequential(
#             conv(n_feats*2, n_feats*4, 1),  # conv1
#             act,
#             conv(n_feats * 4, n_feats, 1),  # conv1
#         )
#
#         self.body2_3 = VisionEncoder(img_dim=args.patch_size,
#                                      patch_dim=args.patch_dim,
#                                      num_channels=n_feats,
#                                      embedding_dim=n_feats * args.patch_dim * args.patch_dim,
#                                      num_heads=args.num_heads,
#                                      num_layers=1,
#                                      hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
#                                      dropout_rate=args.dropout_rate,
#                                      mlp=True,
#                                      pos_every=args.pos_every,
#                                      no_pos=args.no_pos,
#                                      no_norm=args.no_norm,
#                                      no_residual=args.no_residual
#                                      )
#
#         self.body3_1 = nn.Sequential(
#             conv(n_feats, n_feats, kernel_size),   # conv1
#             act,
#             VisionEncoder(img_dim=args.patch_size,
#                           patch_dim=args.patch_dim,
#                           num_channels=n_feats,
#                           embedding_dim=n_feats * args.patch_dim * args.patch_dim,
#                           num_heads=args.num_heads,
#                           num_layers=2,
#                           hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
#                           dropout_rate=args.dropout_rate,
#                           mlp=True,
#                           pos_every=args.pos_every,
#                           no_pos=args.no_pos,
#                           no_norm=args.no_norm,
#                           no_residual=args.no_residual
#                           )
#         )
#
#         self.fusion3_1 = nn.Sequential(
#             conv(n_feats*3, n_feats*4, 1),  # conv1
#             act,
#             conv(n_feats * 4, n_feats, 1),  # conv1
#         )
#
#         self.body3_2 = nn.Sequential(
#             conv(n_feats, n_feats, kernel_size),   # conv1
#             act,
#             VisionEncoder(img_dim=args.patch_size,
#                           patch_dim=args.patch_dim,
#                           num_channels=n_feats,
#                           embedding_dim=n_feats * args.patch_dim * args.patch_dim,
#                           num_heads=args.num_heads,
#                           num_layers=2,
#                           hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
#                           dropout_rate=args.dropout_rate,
#                           mlp=True,
#                           pos_every=args.pos_every,
#                           no_pos=args.no_pos,
#                           no_norm=args.no_norm,
#                           no_residual=args.no_residual
#                           )
#         )
#
#         self.fusion3_2 = nn.Sequential(
#             conv(n_feats*3, n_feats*4, 1),  # conv1
#             act,
#             conv(n_feats * 4, n_feats, 1),  # conv1
#         )
#
#         self.body3_3 = VisionEncoder(img_dim=args.patch_size,
#                                      patch_dim=args.patch_dim,
#                                      num_channels=n_feats,
#                                      embedding_dim=n_feats * args.patch_dim * args.patch_dim,
#                                      num_heads=args.num_heads,
#                                      num_layers=1,
#                                      hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
#                                      dropout_rate=args.dropout_rate,
#                                      mlp=args.no_mlp,
#                                      pos_every=args.pos_every,
#                                      no_pos=True,
#                                      no_norm=args.no_norm,
#                                      no_residual=args.no_residual
#                                      )
#
#         self.fusion3_3 = nn.Sequential(
#             conv(n_feats*3, n_feats*4, 1),  # conv1
#             act,
#             conv(n_feats * 4, n_feats, 1),  # conv1
#         )
#
#         self.body3_4 = VisionEncoder(img_dim=args.patch_size,
#                                      patch_dim=args.patch_dim,
#                                      num_channels=n_feats,
#                                      embedding_dim=n_feats * args.patch_dim * args.patch_dim,
#                                      num_heads=args.num_heads,
#                                      num_layers=1,
#                                      hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
#                                      dropout_rate=args.dropout_rate,
#                                      mlp=args.no_mlp,
#                                      pos_every=args.pos_every,
#                                      no_pos=True,
#                                      no_norm=args.no_norm,
#                                      no_residual=args.no_residual
#                                      )
#
#         self.tail = conv(n_feats, args.n_colors, kernel_size)
#
#     def forward(self, x):
#         y = x
#
#         x = self.head1(x)
#         x = x + self.head1_1(x)
#         x = self.head1_3(x)
#
#         group1 = self.body1_1(x)
#         group2 = self.body2_1(x)
#         group3 = self.body3_1(x)
#
#         group3 = self.body3_2(self.fusion3_1(torch.cat((group1, group2, group3), 1)))
#         group2 = self.body2_2(self.fusion2_1(torch.cat((group1, group2), 1)))
#         group1 = self.body1_2(group1)
#
#         group3 = self.body3_3(self.fusion3_2(torch.cat((group1, group2, group3), 1)))
#         group2 = self.body2_3(self.fusion2_2(torch.cat((group1, group2), 1)))
#
#         group3 = self.body3_4(self.fusion3_3(torch.cat((group1, group2, group3), 1)))
#
#         x = group3
#
#         out = self.tail(x)
#
#         return y - out
#
#
# class VisionTransformer(nn.Module):
#     def __init__(
#             self,
#             img_dim,
#             patch_dim,
#             num_channels,
#             embedding_dim,
#             num_heads,
#             num_layers,
#             hidden_dim,
#             num_queries,
#             positional_encoding_type="learned",
#             dropout_rate=0,
#             no_norm=False,
#             mlp=False,
#             pos_every=False,
#             no_pos=False,
#             no_residual=False
#     ):
#         super(VisionTransformer, self).__init__()
#
#         assert embedding_dim % num_heads == 0
#         assert img_dim % patch_dim == 0
#         self.no_norm = no_norm
#         self.mlp = mlp
#         self.embedding_dim = embedding_dim
#         self.num_heads = num_heads
#         self.patch_dim = patch_dim
#         self.num_channels = num_channels
#
#         self.img_dim = img_dim
#         self.pos_every = pos_every
#         self.num_patches = int((img_dim // patch_dim) ** 2)  # (W*H)/P^2
#         self.seq_length = self.num_patches
#         self.flatten_dim = patch_dim * patch_dim * num_channels  # C*P^2
#
#         self.out_dim = patch_dim * patch_dim * num_channels
#
#         self.no_pos = no_pos
#         self.no_residual = no_residual
#
#         if self.mlp == False:
#             self.linear_encoding = nn.Linear(self.flatten_dim, embedding_dim)
#             self.mlp_head = nn.Sequential(
#                 nn.Linear(embedding_dim, hidden_dim),
#                 nn.Dropout(dropout_rate),
#                 nn.ReLU(),
#                 nn.Linear(hidden_dim, self.out_dim),
#                 nn.Dropout(dropout_rate)
#             )
#
#             self.query_embed = nn.Embedding(num_queries, embedding_dim * self.seq_length)
#             # embedding_dim * self.seq_length ????
#
#         encoder_layer = TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim, dropout_rate, self.no_norm)
#         self.encoder = TransformerEncoder(encoder_layer, num_layers, self.no_residual)
#
#         decoder_layer = TransformerDecoderLayer(embedding_dim, num_heads, hidden_dim, dropout_rate, self.no_norm)
#         self.decoder = TransformerDecoder(decoder_layer, num_layers, self.no_residual)
#
#         if not self.no_pos:
#             self.position_encoding = LearnedPositionalEncoding(
#                 self.seq_length, self.embedding_dim, self.seq_length
#             )
#
#         self.dropout_layer1 = nn.Dropout(dropout_rate)
#
#         if no_norm:
#             for m in self.modules():
#                 if isinstance(m, nn.Linear):
#                     nn.init.normal_(m.weight, std=1 / m.weight.size(1))
#
#     def forward(self, x, query_idx=0, con=False):
#
#         x = torch.nn.functional.unfold(x, self.patch_dim, stride=self.patch_dim).transpose(1, 2).transpose(0, 1).contiguous()  # shape == (time, B, d_model)
#
#         if self.mlp == False:
#             x = self.dropout_layer1(self.linear_encoding(x)) + x
#
#             query_embed = self.query_embed.weight[query_idx].view(-1, 1, self.embedding_dim).repeat(1, x.size(1), 1)
#         else:
#             query_embed = None
#
#         if not self.no_pos:
#             pos = self.position_encoding(x).transpose(0, 1)
#
#         if self.pos_every:
#             x = self.encoder(x, pos=pos)
#             x = self.decoder(x, x, pos=pos, query_pos=query_embed)
#         elif self.no_pos:
#             x = self.encoder(x)
#             x = self.decoder(x, x, query_pos=query_embed)
#         else:
#             x = self.encoder(x + pos)
#             x = self.decoder(x, x, query_pos=query_embed)
#
#         if self.mlp == False:
#             x = self.mlp_head(x) + x
#
#         x = x.transpose(0, 1).contiguous().view(x.size(1), -1, self.flatten_dim)
#
#         if con:
#             con_x = x
#             x = torch.nn.functional.fold(x.transpose(1, 2).contiguous(), int(self.img_dim), self.patch_dim,
#                                          stride=self.patch_dim)
#             return x, con_x
#
#         x = torch.nn.functional.fold(x.transpose(1, 2).contiguous(), int(self.img_dim), self.patch_dim,
#                                      stride=self.patch_dim)
#
#         return x
#
#
# class VisionEncoder(nn.Module):
#     def __init__(
#             self,
#             img_dim,
#             patch_dim,
#             num_channels,
#             embedding_dim,
#             num_heads,
#             num_layers,
#             hidden_dim,
#             dropout_rate=0,
#             no_norm=False,
#             mlp=False,
#             pos_every=False,
#             no_pos=False,
#             no_residual=False
#     ):
#         super(VisionEncoder, self).__init__()
#
#         assert embedding_dim % num_heads == 0
#         assert img_dim % patch_dim == 0
#         self.no_norm = no_norm
#         self.mlp = mlp
#         self.embedding_dim = embedding_dim
#         self.num_heads = num_heads
#         self.patch_dim = patch_dim
#         self.num_channels = num_channels
#
#         self.img_dim = img_dim
#         self.pos_every = pos_every
#         self.num_patches = int((img_dim // patch_dim) ** 2)  # (W*H)/P^2
#         self.seq_length = self.num_patches
#         self.flatten_dim = patch_dim * patch_dim * num_channels  # C*P^2
#
#         self.out_dim = patch_dim * patch_dim * num_channels
#
#         self.no_pos = no_pos
#         self.no_residual = no_residual
#
#         if self.mlp == False: # ??????mlp???????????????encoder?????????
#             self.linear_encoding = nn.Linear(self.flatten_dim, embedding_dim)
#             self.mlp_head = nn.Sequential(
#                 nn.Linear(embedding_dim, hidden_dim),
#                 nn.Dropout(dropout_rate),
#                 nn.ReLU(),
#                 nn.Linear(hidden_dim, self.out_dim),
#                 nn.Dropout(dropout_rate)
#             )
#
#         encoder_layer = TransformerEncoderLayer(patch_dim, embedding_dim, num_heads, hidden_dim, dropout_rate, self.no_norm)
#
#         self.encoder = TransformerEncoder(encoder_layer, num_layers, self.no_residual)
#
#
#         if not self.no_pos:
#             self.position_encoding = LearnedPositionalEncoding(
#                 self.seq_length, self.embedding_dim, self.seq_length
#             )
#
#         self.dropout_layer1 = nn.Dropout(dropout_rate)
#
#         if no_norm:
#             for m in self.modules():
#                 if isinstance(m, nn.Linear):
#                     nn.init.normal_(m.weight, std=1 / m.weight.size(1))
#
#     def forward(self, x, con=False, mask=None):
#
#         x = torch.nn.functional.unfold(x, self.patch_dim, stride=self.patch_dim).transpose(1, 2).transpose(0, 1).contiguous()  # shape == (time, B, d_model)
#
#         if self.mlp == False:
#             x = self.dropout_layer1(self.linear_encoding(x)) + x
#             # query_embed = self.query_embed.weight[query_idx].view(-1, 1, self.embedding_dim).repeat(1, x.size(1), 1)
#         else:
#             pass
#             # query_embed = None
#
#         if not self.no_pos:
#             pos = self.position_encoding(x).transpose(0, 1)
#
#         if self.pos_every:
#             x = self.encoder(x, pos=pos, mask=mask)
#             # x = self.decoder(x, x, pos=pos, query_pos=query_embed)
#         elif self.no_pos:
#             x = self.encoder(x, mask)
#             # x = self.decoder(x, x, query_pos=query_embed)
#         else:
#             x = self.encoder(x + pos, mask)
#             # x = self.decoder(x, x, query_pos=query_embed)
#
#         if self.mlp == False:
#             x = self.mlp_head(x) + x
#
#         x = x.transpose(0, 1).contiguous().view(x.size(1), -1, self.flatten_dim)
#
#         if con:
#             con_x = x
#             x = torch.nn.functional.fold(x.transpose(1, 2).contiguous(), int(self.img_dim), self.patch_dim,
#                                          stride=self.patch_dim)
#             return x, con_x
#
#         x = torch.nn.functional.fold(x.transpose(1, 2).contiguous(), int(self.img_dim), self.patch_dim,
#                                      stride=self.patch_dim)
#
#         return x
#
#
# class LearnedPositionalEncoding(nn.Module):
#     def __init__(self, max_position_embeddings, embedding_dim, seq_length):
#         super(LearnedPositionalEncoding, self).__init__()
#         self.pe = nn.Embedding(max_position_embeddings, embedding_dim)
#         self.seq_length = seq_length
#
#         self.register_buffer(
#             "position_ids", torch.arange(self.seq_length).expand((1, -1))
#         )
#
#     def forward(self, x, position_ids=None):
#         if position_ids is None:
#             position_ids = self.position_ids[:, : self.seq_length]  # self.position_ids???????
#
#         position_embeddings = self.pe(position_ids)
#         return position_embeddings
#
#
# class TransformerEncoder(nn.Module):
#
#     def __init__(self, encoder_layer, num_layers, no_residual=False):
#         super().__init__()
#         self.layers = _get_clones(encoder_layer, num_layers)
#         self.num_layers = num_layers
#         self.no_residual = no_residual
#
#     def forward(self, src, pos=None, mask=None):
#         output = src
#
#         if self.no_residual or len(self.layers) < 4:
#             for layer in self.layers:
#                 output = layer(output, pos=pos, mask=mask)
#         else:  # encoder use residual struct
#             layers = iter(self.layers)
#             # ????????? ?????? ????????????
#             output1 = next(layers)(output, pos=pos, mask=mask)
#             output2 = next(layers)(output1, pos=pos, mask=mask)
#             output3 = next(layers)(output2, pos=pos, mask=mask)
#             output4 = next(layers)(output3, pos=pos, mask=mask)
#             output = output + output1 + output2 + output3 + output4
#
#             for layer in layers:
#                 output = layer(output, pos=pos, mask=mask)
#
#         return output
#
#
# class TransformerEncoderLayer(nn.Module):
#     def __init__(self, patch_dim, d_model, nhead, dim_feedforward=2048, dropout=0.1, no_norm=False,
#                  activation="relu"):
#         super().__init__()
#         self.patch_dim = patch_dim
#
#         # multihead attention
#         self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
#
#         # Implementation of Feedforward model
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)
#
#         self.norm1 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
#         self.norm2 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#
#         self.activation = _get_activation_fn(activation)
#
#         nn.init.kaiming_uniform_(self.self_attn.in_proj_weight, a=math.sqrt(5))
#
#     def with_pos_embed(self, tensor, pos):
#         return tensor if pos is None else tensor + pos
#
#     def forward(self, src, pos=None, mask=None):
#         src2 = self.norm1(src)
#         q = k = self.with_pos_embed(src2, pos)
#
#         src2 = self.self_attn(q, k, src2)
#
#         src = src + self.dropout1(src2[0])
#         src2 = self.norm2(src)
#         src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
#         src = src + self.dropout2(src2)
#         return src
#
#
# class TransformerDecoder(nn.Module):
#
#     def __init__(self, decoder_layer, num_layers, no_residual=False):
#         super().__init__()
#         self.layers = _get_clones(decoder_layer, num_layers)
#         self.num_layers = num_layers
#
#     def forward(self, tgt, memory, pos=None, query_pos=None):
#         output = tgt
#
#         for layer in self.layers:
#             output = layer(output, memory, pos=pos, query_pos=query_pos)
#
#         return output
#
#
# class TransformerDecoderLayer(nn.Module):
#
#     def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, no_norm=False,
#                  activation="relu"):
#         super().__init__()
#         self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
#         self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
#         # Implementation of Feedforward model
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)
#
#         self.norm1 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
#         self.norm2 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
#         self.norm3 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.dropout3 = nn.Dropout(dropout)
#
#         self.activation = _get_activation_fn(activation)
#
#     def with_pos_embed(self, tensor, pos):
#         return tensor if pos is None else tensor + pos
#
#     def forward(self, tgt, memory, pos=None, query_pos=None):
#         tgt2 = self.norm1(tgt)
#         q = k = self.with_pos_embed(tgt2, query_pos)
#         tgt2 = self.self_attn(q, k, value=tgt2)[0]  # [0]
#         tgt = tgt + self.dropout1(tgt2)
#         tgt2 = self.norm2(tgt)
#         tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
#                                    key=self.with_pos_embed(memory, pos),
#                                    value=memory)[0]
#         tgt = tgt + self.dropout2(tgt2)
#         tgt2 = self.norm3(tgt)
#         tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
#         tgt = tgt + self.dropout3(tgt2)
#         return tgt
#
#
# def _get_clones(module, N):
#     return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
#
#
# def _get_activation_fn(activation):
#     """Return an activation function given a string"""
#     if activation == "relu":
#         return F.relu
#     if activation == "gelu":
#         return F.gelu
#     if activation == "glu":
#         return F.glu
#     raise RuntimeError(F"activation should be relu/gelu, not {activation}.")