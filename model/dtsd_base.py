'''
    Dynamic_conv + transformer + ssim + Denosing
'''

import sys

import torch

sys.path.append('../')

from model_common import common
from torch import nn
from model_common.transformer_module import VisionEncoder
from model_common.WB_TransformerLayer import WBformerLayer, CBformerLayer, Dual_WBformaerlayer
from model_common.wbf import WBF
import torch.nn.functional as F

def make_model(args):
    return DTSD(args), 1


class DTSD(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(DTSD, self).__init__()

        self.scale_idx = 0

        self.args = args

        n_feats = args.n_feats
        kernel_size = 3
        act = nn.ReLU(True)

        self.sub_mean = common.MeanShift(args.rgb_range)  # sub = 减
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)  # add = 加
        if self.args.flag == 0:
            self.head = nn.Sequential(
                    conv(args.n_colors, n_feats, kernel_size),  # conv1
                    common.ResBlock(conv, n_feats, 5, act=act),  # conv2
                    common.ResBlock(conv, n_feats, 5, act=act),  # conv3
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
            self.head = nn.Sequential(
                conv(args.n_colors, n_feats, kernel_size),  # conv1
                common.ResBlock(conv, n_feats, 5, act=act),  # conv2
                common.ResBlock(conv, n_feats, 5, act=act),  # conv3
            )

            self.body1 = VisionEncoder(img_dim=args.patch_size,
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

            self.body2 = VisionEncoder(img_dim=args.patch_size,
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


            self.tail = conv(n_feats*2, args.n_colors, kernel_size)

        if self.args.flag == 2:
            self.head = nn.Sequential(
                conv(args.n_colors, n_feats, kernel_size),  # conv1
                common.ResBlock(conv, n_feats, 5, act=act),  # conv2
                common.ResBlock(conv, n_feats, 5, act=act),  # conv3
            )

            self.patch_size_base = int(args.patch_size/3) # 48

            self.patch_size1 = self.patch_size_base*3
            self.patch_dim1 = args.patch_dim*3

            self.stage1 = VisionEncoder(img_dim=self.patch_size1,
                                       patch_dim=self.patch_dim1,
                                       num_channels=n_feats,
                                       embedding_dim=n_feats * self.patch_dim1 * self.patch_dim1,
                                       num_heads=args.num_heads,
                                       num_layers=1,
                                       hidden_dim=n_feats * self.patch_dim1 * self.patch_dim1 * 4,
                                       dropout_rate=args.dropout_rate,
                                       mlp=args.no_mlp,
                                       pos_every=args.pos_every,
                                       no_pos=args.no_pos,
                                       no_norm=args.no_norm,
                                       no_residual=args.no_residual
                                       )

            self.patch_size2 = self.patch_size_base*2
            self.patch_dim2 = args.patch_dim*2

            self.stage2 = VisionEncoder(img_dim=self.patch_size2,
                                       patch_dim=self.patch_dim2,
                                       num_channels=n_feats,
                                       embedding_dim=n_feats * self.patch_dim2 * self.patch_dim2,
                                       num_heads=args.num_heads,
                                       num_layers=5,
                                       hidden_dim=n_feats * self.patch_dim2 * self.patch_dim2 * 4,
                                       dropout_rate=args.dropout_rate,
                                       mlp=args.no_mlp,
                                       pos_every=args.pos_every,
                                       no_pos=args.no_pos,
                                       no_norm=args.no_norm,
                                       no_residual=args.no_residual
                                       )

            self.patch_size3 = self.patch_size_base
            self.patch_dim3 = args.patch_dim

            self.stage3 = VisionEncoder(img_dim=self.patch_size3,
                                        patch_dim=self.patch_dim3,
                                        num_channels=n_feats,
                                        embedding_dim=n_feats * self.patch_dim3 * self.patch_dim3,
                                        num_heads=args.num_heads,
                                        num_layers=1,
                                        hidden_dim=n_feats * self.patch_dim3 * self.patch_dim3 * 4,
                                        dropout_rate=args.dropout_rate,
                                        mlp=args.no_mlp,
                                        pos_every=args.pos_every,
                                        no_pos=args.no_pos,
                                        no_norm=args.no_norm,
                                        no_residual=args.no_residual
                                        )
            self.tail = conv(n_feats, args.n_colors, kernel_size)

        if self.args.flag == 3:
            self.head = nn.Sequential(
                conv(args.n_colors, n_feats, kernel_size),  # conv1
                common.ResBlock(conv, n_feats, 5, act=act),  # conv2
                common.ResBlock(conv, n_feats, 5, act=act),  # conv3
            )

            self.patch_size_base = int(args.patch_size/3)  # 32

            self.patch_size1 = self.patch_size_base*2
            self.patch_dim1 = args.patch_dim*2

            self.stage1 = VisionEncoder(img_dim=self.patch_size1,
                                       patch_dim=self.patch_dim1,
                                       num_channels=n_feats,
                                       embedding_dim=n_feats * self.patch_dim1 * self.patch_dim1,
                                       num_heads=args.num_heads,
                                       num_layers=2,
                                       hidden_dim=n_feats * self.patch_dim1 * self.patch_dim1 * 4,
                                       dropout_rate=args.dropout_rate,
                                       mlp=args.no_mlp,
                                       pos_every=args.pos_every,
                                       no_pos=args.no_pos,
                                       no_norm=args.no_norm,
                                       no_residual=args.no_residual
                                       )

            self.patch_size2 = self.patch_size_base
            self.patch_dim2 = args.patch_dim

            self.stage2 = VisionEncoder(img_dim=self.patch_size2,
                                       patch_dim=self.patch_dim2,
                                       num_channels=n_feats,
                                       embedding_dim=n_feats * self.patch_dim2 * self.patch_dim2,
                                       num_heads=args.num_heads,
                                       num_layers=args.num_layers,
                                       hidden_dim=n_feats * self.patch_dim2 * self.patch_dim2 * 4,
                                       dropout_rate=args.dropout_rate,
                                       mlp=args.no_mlp,
                                       pos_every=args.pos_every,
                                       no_pos=args.no_pos,
                                       no_norm=args.no_norm,
                                       no_residual=args.no_residual
                                       )

            self.tail = conv(n_feats, args.n_colors, kernel_size)

        if self.args.flag == 4:
            self.head = nn.Sequential(
                conv(args.n_colors, n_feats, kernel_size),  # conv1
                common.ResBlock(conv, n_feats, 5, act=act),  # conv2
                common.ResBlock(conv, n_feats, 5, act=act),  # conv3
            )

            self.patch_size_base = int(args.patch_size/3)  # 32

            self.patch_size1 = self.patch_size_base*2
            self.patch_dim1 = args.patch_dim*2
            n_heads = self.args.num_heads*2

            self.stage1 = VisionEncoder(img_dim=self.patch_size1,
                                       patch_dim=self.patch_dim1,
                                       num_channels=n_feats,
                                       embedding_dim=n_feats * self.patch_dim1 * self.patch_dim1,
                                       num_heads=n_heads,
                                       num_layers=2,
                                       hidden_dim=n_feats * self.patch_dim1 * self.patch_dim1 * 4,
                                       dropout_rate=args.dropout_rate,
                                       mlp=args.no_mlp,
                                       pos_every=args.pos_every,
                                       no_pos=args.no_pos,
                                       no_norm=args.no_norm,
                                       no_residual=args.no_residual
                                       )

            self.patch_size2 = int(self.patch_size_base*3/2)
            self.patch_dim2 = int(args.patch_dim*3/2)
            n_heads = int(args.num_heads*3/2)

            self.stage2 = VisionEncoder(img_dim=self.patch_size2,
                                       patch_dim=self.patch_dim2,
                                       num_channels=n_feats,
                                       embedding_dim=n_feats * self.patch_dim2 * self.patch_dim2,
                                       num_heads=n_heads,
                                       num_layers=args.num_layers,
                                       hidden_dim=n_feats * self.patch_dim2 * self.patch_dim2 * 4,
                                       dropout_rate=args.dropout_rate,
                                       mlp=args.no_mlp,
                                       pos_every=args.pos_every,
                                       no_pos=args.no_pos,
                                       no_norm=args.no_norm,
                                       no_residual=args.no_residual
                                       )

            self.tail = conv(n_feats, args.n_colors, kernel_size)

        if self.args.flag == 5:
            self.head = nn.Sequential(
                conv(args.n_colors, n_feats, kernel_size),  # conv1
                common.ResBlock(conv, n_feats, 5, act=act),  # conv2
                common.ResBlock(conv, n_feats, 5, act=act),  # conv3
            )

            self.patch_size_base = int(args.patch_size/3)  # 32

            self.patch_size2 = int(self.patch_size_base*3/2)
            self.patch_dim2 = int(args.patch_dim*3/2)
            n_heads = int(args.num_heads*3/2)

            self.stage2 = VisionEncoder(img_dim=self.patch_size2,
                                       patch_dim=self.patch_dim2,
                                       num_channels=n_feats,
                                       embedding_dim=n_feats * self.patch_dim2 * self.patch_dim2,
                                       num_heads=n_heads,
                                       num_layers=args.num_layers,
                                       hidden_dim=n_feats * self.patch_dim2 * self.patch_dim2 * 4,
                                       dropout_rate=args.dropout_rate,
                                       mlp=args.no_mlp,
                                       pos_every=args.pos_every,
                                       no_pos=args.no_pos,
                                       no_norm=args.no_norm,
                                       no_residual=args.no_residual
                                       )

            self.tail = conv(n_feats, args.n_colors, kernel_size)

        if self.args.flag == 6:
            self.head = nn.Sequential(
                conv(args.n_colors, n_feats, kernel_size),  # conv1
                common.ResBlock(conv, n_feats, 5, act=act),  # conv2
                common.ResBlock(conv, n_feats, 5, act=act),  # conv3
            )

            self.patch_size_base = int(args.patch_size/3)  # 32

            self.patch_size1 = self.patch_size_base*2
            self.patch_dim1 = args.patch_dim*2
            n_heads = self.args.num_heads*2

            self.stage1 = VisionEncoder(img_dim=self.patch_size1,
                                       patch_dim=self.patch_dim1,
                                       num_channels=n_feats,
                                       embedding_dim=n_feats * self.patch_dim1 * self.patch_dim1,
                                       num_heads=n_heads,
                                       num_layers=2,
                                       hidden_dim=n_feats * self.patch_dim1 * self.patch_dim1 * 4,
                                       dropout_rate=args.dropout_rate,
                                       mlp=args.no_mlp,
                                       pos_every=args.pos_every,
                                       no_pos=args.no_pos,
                                       no_norm=args.no_norm,
                                       no_residual=args.no_residual
                                       )

            self.patch_size2 = int(self.patch_size_base*3/2)
            self.patch_dim2 = int(args.patch_dim*3/2)
            n_heads = int(args.num_heads*3/2)

            self.stage2 = VisionEncoder(img_dim=self.patch_size2,
                                       patch_dim=self.patch_dim2,
                                       num_channels=n_feats,
                                       embedding_dim=n_feats * self.patch_dim2 * self.patch_dim2,
                                       num_heads=n_heads,
                                       num_layers=args.num_layers,
                                       hidden_dim=n_feats * self.patch_dim2 * self.patch_dim2 * 4,
                                       dropout_rate=args.dropout_rate,
                                       mlp=args.no_mlp,
                                       pos_every=args.pos_every,
                                       no_pos=args.no_pos,
                                       no_norm=args.no_norm,
                                       no_residual=args.no_residual
                                       )

            self.tail = conv(n_feats, args.n_colors, kernel_size)

        if self.args.flag == 7:

            self.head = nn.Sequential(
                conv(args.n_colors, n_feats, kernel_size),  # conv1
                common.ResBlock(conv, n_feats, 5, act=act),  # conv2
                common.ResBlock(conv, n_feats, 5, act=act),  # conv3
            )

            self.patch_dim = args.patch_dim
            self.img_size = args.patch_size
            self.num_heads = args.num_heads
            self.window_size = int(args.patch_size/2)
            self.num_layers = args.num_layers
            self.mlp_ratio = 4

            drop_path_rate = 0.1
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_layers)]  # stochastic depth decay rule

            layers = ['no_shift',
                      'horizontal', 'vertical', 'no_shift',
                      'horizontal', 'vertical', 'no_shift',
                      'horizontal', 'vertical', 'no_shift'
                      ]

            assert self.num_layers == len(layers)
            '''
            self.body = nn.Sequential(
                WBformerLayer(dim=n_feats, img_size=self.img_size, num_heads=self.num_heads, patch_dim=self.patch_dim,
                              window_size=self.window_size, layers=layers, drop_path=dpr, mlp_ratio=self.mlp_ratio, ape=True)
            )
            '''

            self.body = nn.Sequential(
                WBF(dim=n_feats, input_szie=self.img_size, num_heads=self.num_heads, window_size=self.window_size,
                    layer=layers, patch_dim=self.patch_dim, mlp_ratio=self.mlp_ratio, drop_path=dpr, ape=True)
            )

            self.tail = conv(n_feats, args.n_colors, kernel_size)

        if self.args.flag == 8:
            self.head = nn.Sequential(
                conv(args.n_colors, n_feats, kernel_size),  # conv1
                common.ResBlock(conv, n_feats, 5, act=act),  # conv2
                common.ResBlock(conv, n_feats, 5, act=act),  # conv3
            )

            self.patch_dim = args.patch_dim
            self.img_size = args.patch_size
            self.num_heads = args.num_heads
            self.window_size = int(args.patch_size/2)
            self.num_layers = args.num_layers
            self.mlp_ratio = 4

            drop_path_rate = 0.0
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_layers)]  # stochastic depth decay rule

            layers = ['no_shift', 'no_shift', 'no_shift', 'no_shift', 'no_shift', 'no_shift',
                      'no_shift', 'no_shift', 'no_shift', 'no_shift']

            assert self.num_layers == len(layers)

            '''
            self.body = nn.Sequential(
                WBformerLayer(dim=n_feats, img_size=self.img_size, num_heads=self.num_heads, patch_dim=self.patch_dim,
                              window_size=self.window_size, layers=layers, drop_path=dpr, mlp_ratio=self.mlp_ratio, ape=True)
            )
            '''

            self.body = nn.Sequential(
                WBF(dim=n_feats, input_szie=self.img_size, num_heads=self.num_heads, window_size=self.window_size,
                    layer=layers, patch_dim=self.patch_dim, mlp_ratio=self.mlp_ratio, drop_path=dpr, ape=True)
            )

            self.tail = conv(n_feats, args.n_colors, kernel_size)

        if self.args.flag == 9:

            self.head = nn.Sequential(
                conv(args.n_colors, n_feats, kernel_size),  # conv1
                common.ResBlock(conv, n_feats, 5, act=act),  # conv2
                common.ResBlock(conv, n_feats, 5, act=act),  # conv3
            )

            self.patch_dim = args.patch_dim
            self.img_size = args.patch_size
            self.num_heads = args.num_heads
            self.window_size = int(args.patch_size/2)
            self.num_layers = args.num_layers

            drop_path_rate = 0.1
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_layers)]  # stochastic depth decay rule


            layers = [
                      'horizontal', 'vertical', 'no_shift',
                      'horizontal', 'vertical', 'no_shift',
                      'horizontal', 'vertical'
                      ]

            assert self.num_layers == len(layers)

            self.body1 = CBformerLayer(dim=n_feats, img_size=self.img_size, num_heads=self.num_heads*4,
                                       patch_dim=self.patch_dim*2, kernel_size=5, ape=True)
            self.body2 = nn.Sequential(
                WBformerLayer(dim=n_feats, img_size=self.img_size, num_heads=self.num_heads, patch_dim=self.patch_dim,
                              window_size=self.window_size, layers=layers, drop_path=dpr, mlp_ratio=4., ape=False)
            )

            self.body3 = CBformerLayer(dim=n_feats, img_size=self.img_size, num_heads=self.num_heads*4,
                                       patch_dim=self.patch_dim * 2, kernel_size=5, ape=False)

            self.tail = conv(n_feats, args.n_colors, kernel_size)

        if self.args.flag == 10:

            self.head = nn.Sequential(
                conv(args.n_colors, n_feats, kernel_size),  # conv1
                common.ResBlock(conv, n_feats, 5, act=act),  # conv2
                common.ResBlock(conv, n_feats, 5, act=act),  # conv3
            )

            assert args.patch_size % 2 == 0 and args.patch_size % 3 == 0

            self.patch_dim = args.patch_dim
            self.patch_dim2 = 2
            self.img_size = args.patch_size
            self.num_heads = args.num_heads
            self.window_size = int(args.patch_size / 2)
            self.num_layers = args.num_layers

            drop_path_rate = 0.1
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_layers)]  # stochastic depth decay rule

            layers = [
                      'horizontal', 'vertical', 'no_shift',
                      'horizontal', 'vertical', 'no_shift',
                      'horizontal', 'vertical', 'no_shift'
                      ]

            assert self.num_layers == len(layers)

            self.body1 = CBformerLayer(dim=n_feats, img_size=self.img_size, num_heads=self.num_heads*4,
                                       patch_dim=self.patch_dim*2, kernel_size=5, ape=True)

            self.body2 = WBformerLayer(dim=n_feats, img_size=self.img_size, num_heads=self.num_heads,
                                       patch_dim=self.patch_dim, window_size=self.window_size, layers=layers[0:3],
                                       drop_path=dpr[0:3], mlp_ratio=4., ape=False)

            self.body3 = Dual_WBformaerlayer(dim=n_feats, img_size=self.img_size,
                                             layers1=layers[3:6],
                                             layers2=layers[6:],
                                             num_heads1=self.num_heads,
                                             num_heads2=16,
                                             patch_dim1=self.patch_dim,
                                             patch_dim2=self.patch_dim2,
                                             drop_path1=dpr[3:][1::2],
                                             drop_path2=dpr[3:][::2],
                                             mlp_ratio=4., ape=False)

            self.body4 = CBformerLayer(dim=n_feats, img_size=self.img_size, num_heads=self.num_heads * 4,
                                       patch_dim=self.patch_dim * 2, kernel_size=5, ape=False)

            self.tail = conv(n_feats, args.n_colors, kernel_size)

        if self.args.flag == 11:
            self.head = nn.Sequential(
                conv(args.n_colors, n_feats, kernel_size),  # conv1
                common.ResBlock(conv, n_feats, 5, act=act),  # conv2
                common.ResBlock(conv, n_feats, 5, act=act),  # conv3
            )

            assert args.patch_size % 2 == 0 and args.patch_size % 3 == 0

            self.patch_dim = args.patch_dim
            self.patch_dim1 = 2
            self.img_size = args.patch_size
            self.num_heads = args.num_heads
            self.num_heads1 = 16
            self.window_size = int(args.patch_size / 2)
            self.window_size1 = int(args.patch_size / 3)
            self.num_layers = args.num_layers

            drop_path_rate = 0.1
            dpr = [x.item() for x in
                   torch.linspace(0, drop_path_rate, self.num_layers)]  # stochastic depth decay rule

            layers = [
                'horizontal', 'vertical', 'no_shift',
                'horizontal', 'vertical', 'no_shift',
                'horizontal', 'vertical', 'no_shift',
            ]

            assert self.num_layers == len(layers)

            self.body1 = CBformerLayer(dim=n_feats, img_size=self.img_size, num_heads=self.num_heads * 4,
                                       patch_dim=self.patch_dim * 2, kernel_size=5, ape=True)

            self.body2 = WBformerLayer(dim=n_feats, img_size=self.img_size, num_heads=self.num_heads,
                                       patch_dim=self.patch_dim,
                                       window_size=self.window_size, layers=layers[0:3], drop_path=dpr[0:3],
                                       mlp_ratio=4., ape=False)

            self.body3 = WBformerLayer(dim=n_feats, img_size=self.img_size, num_heads=self.num_heads1,
                                       patch_dim=self.patch_dim1,
                                       window_size=self.window_size1, layers=layers[3:6], drop_path=dpr[3:6],
                                       mlp_ratio=4., ape=False)

            self.body4 = WBformerLayer(dim=n_feats, img_size=self.img_size, num_heads=self.num_heads,
                                       patch_dim=self.patch_dim,
                                       window_size=self.window_size, layers=layers[6:], drop_path=dpr[6:],
                                       mlp_ratio=4., ape=False)

            self.body5 = CBformerLayer(dim=n_feats, img_size=self.img_size, num_heads=self.num_heads * 4,
                                       patch_dim=self.patch_dim * 2, kernel_size=5, ape=False)

            self.tail = conv(n_feats, args.n_colors, kernel_size)

        if self.args.flag == 12:
            self.head = nn.Sequential(
                conv(args.n_colors, n_feats, kernel_size),  # conv1
                common.ResBlock(conv, n_feats, 5, act=act),  # conv2
                common.ResBlock(conv, n_feats, 5, act=act),  # conv3
            )

            assert args.patch_size % 2 == 0 and args.patch_size % 3 == 0

            self.patch_dim = args.patch_dim
            self.img_size = args.patch_size
            self.num_heads = args.num_heads
            self.window_size = int(args.patch_size / 2)
            self.num_layers = args.num_layers

            drop_path_rate = 0.1
            dpr = [x.item() for x in
                   torch.linspace(0, drop_path_rate, self.num_layers)]  # stochastic depth decay rule

            layers = [
                'horizontal', 'no_shift',
                'horizontal', 'no_shift',
                'horizontal', 'no_shift',
                'horizontal', 'no_shift',
                'horizontal', 'no_shift',
            ]

            assert self.num_layers == len(layers)

            self.body = nn.Sequential(
                WBformerLayer(dim=n_feats, img_size=self.img_size, num_heads=self.num_heads, patch_dim=self.patch_dim,
                              window_size=self.window_size, layers=layers, drop_path=dpr, ape=True)
            )

            self.tail = conv(n_feats, args.n_colors, kernel_size)

        if self.args.flag == 13:
            self.head = nn.Sequential(
                conv(args.n_colors, n_feats, kernel_size),  # conv1
                common.ResBlock(conv, n_feats, 5, act=act),  # conv2
                common.ResBlock(conv, n_feats, 5, act=act),  # conv3
            )

            self.tail = conv(n_feats, args.n_colors, kernel_size)

        if self.args.flag == 14:
            self.head = nn.Sequential(
                conv(args.n_colors, n_feats, kernel_size),  # conv1
                common.ResBlock(conv, n_feats, 5, act=act),  # conv2
                common.ResBlock(conv, n_feats, 5, act=act),  # conv3
            )

            self.patch_dim = args.patch_dim
            self.img_size = args.patch_size
            self.num_heads = args.num_heads
            self.window_size = int(args.patch_size / 2)
            self.num_layers = args.num_layers

            drop_path_rate = 0.1
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_layers)]  # stochastic depth decay rule

            layers = ['no_shift',
                      'horizontal', 'vertical', 'no_shift',
                      'horizontal', 'vertical', 'no_shift',
                      'horizontal', 'vertical', 'no_shift'
                      ]

            assert self.num_layers == len(layers)

            self.body = nn.Sequential(
                WBformerLayer(dim=n_feats, img_size=self.img_size, num_heads=self.num_heads, patch_dim=self.patch_dim,
                              window_size=self.window_size, layers=layers, drop_path=dpr, mlp_ratio=4., ape=True)
            )

            self.tail = conv(n_feats, args.n_colors, kernel_size)

        if self.args.flag == 15:
            self.head = nn.Sequential(
                conv(args.n_colors, n_feats, kernel_size),  # conv1
                common.ResBlock(conv, n_feats, 5, act=act),  # conv2
                common.ResBlock(conv, n_feats, 5, act=act),  # conv3
            )

            self.patch_dim = args.patch_dim
            self.img_size = args.patch_size
            self.num_heads = args.num_heads
            self.window_size = int(args.patch_size / 2)
            self.num_layers = args.num_layers
            self.mlp_ratio = 4

            drop_path_rate = 0.0
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_layers)]  # stochastic depth decay rule

            layers = ['no_shift', 'no_shift', 'no_shift', 'no_shift', 'no_shift', 'no_shift',
                      'no_shift', 'no_shift', 'no_shift', 'no_shift']

            assert self.num_layers == len(layers)

            self.body1 = nn.Sequential(
                WBF(dim=n_feats, input_szie=self.img_size, num_heads=self.num_heads, window_size=self.window_size,
                    layer=layers[0:5], patch_dim=self.patch_dim, mlp_ratio=self.mlp_ratio, drop_path=dpr[0:5], ape=True)
            )

            self.body2 = nn.Sequential(
                WBF(dim=n_feats, input_szie=self.img_size, num_heads=self.num_heads, window_size=self.window_size,
                    layer=layers[5:10], patch_dim=self.patch_dim, mlp_ratio=self.mlp_ratio, drop_path=dpr[5:10], ape=False)
            )

            self.tail = conv(n_feats, args.n_colors, kernel_size)

        if self.args.flag == 16:
            self.head = nn.Sequential(
                conv(args.n_colors, n_feats, kernel_size),  # conv1
                common.ResBlock(conv, n_feats, 5, act=act),  # conv2
                common.ResBlock(conv, n_feats, 5, act=act),  # conv3
            )

            self.patch_dim = args.patch_dim
            self.img_size = args.patch_size
            self.num_heads = args.num_heads
            self.window_size = int(args.patch_size / 2)
            self.num_layers = args.num_layers

            drop_path_rate = 0.1
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_layers)]  # stochastic depth decay rule

            layers = ['no_shift',
                      'horizontal', 'vertical', 'no_shift',
                      'horizontal', 'vertical', 'no_shift',
                      'horizontal', 'vertical', 'no_shift'
                      ]

            assert self.num_layers == len(layers)

            self.body = nn.Sequential(
                WBformerLayer(dim=n_feats, img_size=self.img_size, num_heads=self.num_heads, patch_dim=self.patch_dim,
                              window_size=self.window_size, layers=layers, drop_path=dpr, mlp_ratio=4., ape=True)
            )

            self.tail = conv(n_feats, args.n_colors, kernel_size)

        if self.args.flag == 17:
            self.head = nn.Sequential(
                conv(args.n_colors, n_feats, kernel_size),  # conv1
                common.ResBlock(conv, n_feats, 5, act=act),  # conv2
                common.ResBlock(conv, n_feats, 5, act=act),  # conv3
            )

            self.img_size = args.patch_size

            self.patch_dim = args.patch_dim

            self.num_heads = args.num_heads

            self.window_size = int(args.patch_size / 2)

            self.num_layers = args.num_layers

            drop_path_rate = 0.1
            dpr = [x.item() for x in
                   torch.linspace(0, drop_path_rate, self.num_layers)]  # stochastic depth decay rule

            layers = [
                'horizontal', 'vertical', 'no_shift',
                'horizontal', 'vertical',
                'no_shift', 'horizontal', 'vertical'
            ]

            assert self.num_layers == len(layers)

            self.body1 = CBformerLayer(dim=n_feats, img_size=self.img_size, num_heads=self.num_heads * 4,
                                       patch_dim=self.patch_dim * 2, kernel_size=5, ape=True)
            self.body2 = nn.Sequential(
                WBformerLayer(dim=n_feats, img_size=self.img_size, num_heads=self.num_heads,
                              patch_dim=self.patch_dim,
                              window_size=self.window_size, layers=layers[0:3], drop_path=dpr[0:3], mlp_ratio=4.,
                              ape=False)
            )

            self.body3 = nn.Sequential(
                WBformerLayer(dim=n_feats, img_size=self.img_size, num_heads=self.num_heads,
                              patch_dim=self.patch_dim,
                              window_size=self.window_size, layers=layers[3:5], drop_path=dpr[3:5], mlp_ratio=4.,
                              ape=False)
            )

            self.body4 = nn.Sequential(
                WBformerLayer(dim=n_feats, img_size=self.img_size, num_heads=self.num_heads,
                              patch_dim=self.patch_dim,
                              window_size=self.window_size, layers=layers[5:], drop_path=dpr[5:], mlp_ratio=4.,
                              ape=False)
            )

            self.body5 = CBformerLayer(dim=n_feats, img_size=self.img_size, num_heads=self.num_heads * 4,
                                       patch_dim=self.patch_dim * 2, kernel_size=5, ape=False)

            self.tail = conv(n_feats, args.n_colors, kernel_size)

        if self.args.flag == 18:

            self.head = nn.Sequential(
                conv(args.n_colors, n_feats, kernel_size),  # conv1
                common.ResBlock(conv, n_feats, 5, act=act),  # conv2
                common.ResBlock(conv, n_feats, 5, act=act),  # conv3
            )

            self.patch_dim = args.patch_dim
            self.img_size = args.patch_size
            self.num_heads = args.num_heads
            self.window_size = int(args.patch_size/2)
            self.num_layers = args.num_layers

            drop_path_rate = 0.1
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_layers)]  # stochastic depth decay rule

            layers = ['horizontal', 'vertical', 'no_shift',
                      'horizontal', 'vertical',
                      'no_shift', 'horizontal', 'vertical'
                      ]

            assert self.num_layers == len(layers)

            self.body1 = CBformerLayer(dim=n_feats, img_size=self.img_size, num_heads=self.num_heads*4,
                                       patch_dim=self.patch_dim*2, kernel_size=5, ape=True)
            self.body2 = nn.Sequential(
                WBformerLayer(dim=n_feats, img_size=self.img_size, num_heads=self.num_heads, patch_dim=self.patch_dim,
                              window_size=self.window_size, layers=layers, drop_path=dpr, mlp_ratio=4., ape=False,
                              )
            )

            self.body3 = CBformerLayer(dim=n_feats, img_size=self.img_size, num_heads=self.num_heads*4,
                                       patch_dim=self.patch_dim * 2, kernel_size=5, ape=False)

            self.tail = conv(n_feats, args.n_colors, kernel_size)

        if self.args.flag == 19:

            self.head = nn.Sequential(
                conv(args.n_colors, n_feats, kernel_size),  # conv1
                common.ResBlock(conv, n_feats, 5, act=act),  # conv2
                common.ResBlock(conv, n_feats, 5, act=act),  # conv3
            )

            self.patch_dim = args.patch_dim
            self.img_size = args.patch_size
            self.num_heads = args.num_heads
            self.window_size = int(args.patch_size/2)
            self.num_layers = args.num_layers

            drop_path_rate = 0.1
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_layers)]  # stochastic depth decay rule


            layers = [
                      'horizontal', 'vertical', 'no_shift',
                      'horizontal', 'vertical', 'no_shift',
                      'horizontal', 'vertical'
                      ]

            assert self.num_layers == len(layers)

            self.body1 = CBformerLayer(dim=n_feats, img_size=self.img_size, num_heads=self.num_heads*4,
                                       patch_dim=self.patch_dim*2, kernel_size=5, ape=True)

            self.body2 = nn.Sequential(
                WBformerLayer(dim=n_feats, img_size=self.img_size, num_heads=self.num_heads, patch_dim=self.patch_dim,
                              window_size=self.window_size, layers=layers, drop_path=dpr, mlp_ratio=4., ape=False)
            )

            self.body3 = CBformerLayer(dim=n_feats, img_size=self.img_size, num_heads=self.num_heads*4,
                                       patch_dim=self.patch_dim * 2, kernel_size=5, ape=False)

            self.tail = conv(n_feats, args.n_colors, kernel_size)

        if self.args.flag == 20:
            self.head = nn.Sequential(
                conv(args.n_colors, n_feats, kernel_size),  # conv1
                common.ResBlock(conv, n_feats, 5, act=act),  # conv2
                common.ResBlock(conv, n_feats, 5, act=act),  # conv3
            )

            self.patch_dim = args.patch_dim
            self.img_size = args.patch_size
            self.num_heads = args.num_heads

            self.body1 = CBformerLayer(dim=n_feats, img_size=self.img_size, num_heads=self.num_heads * 4,
                                       patch_dim=self.patch_dim * 2, kernel_size=5, ape=True)

            self.body2 = CBformerLayer(dim=n_feats, img_size=self.img_size, num_heads=self.num_heads * 4,
                                       patch_dim=self.patch_dim * 2, kernel_size=5, ape=False)

            self.tail = conv(n_feats, args.n_colors, kernel_size)

    def forward(self, x):
        y = x

        x = self.head(x)

        if self.args.flag == 0:
            x = self.body(x)
        elif self.args.flag == 1:
            x = torch.cat((self.body1(x), self.body2(x)), 1)
        elif self.args.flag == 2:
            bs = x.size(0)  # batch_size

            x = self.stage1(x)

            # x_unfold = F.unfold(x, self.patch_size2, stride=self.patch_size_base).transpose(0, 2).contigous()
            x_unfold = F.unfold(x, self.patch_size2, stride=self.patch_size_base).transpose(0, 2).contiguous()
            x_unfold = x_unfold.view(x_unfold.size(0), -1, self.patch_size2, self.patch_size2)
            y_unfold = []
            for i in range(x_unfold.size(0)):
                y_unfold.append(
                    self.stage2(x_unfold[i:i+1, ...].view(bs, -1, self.patch_size2, self.patch_size2))\
                        .view(1, -1, self.patch_size2, self.patch_size2)
                )
            y_unfold = torch.cat(y_unfold, dim=0)
            x = F.fold(y_unfold.view(y_unfold.size(0), -1, bs).transpose(0, 2).contiguous(),
                       (self.patch_size1, self.patch_size1), kernel_size=self.patch_size2,
                       stride=self.patch_size_base)

            x_unfold = F.unfold(x, self.patch_size3, stride=self.patch_size_base).transpose(0, 2).contiguous()
            x_unfold = x_unfold.view(x_unfold.size(0), -1, self.patch_size3, self.patch_size3)
            y_unfold = []
            for i in range(x_unfold.size(0)):
                y_unfold.append(
                    self.stage3(x_unfold[i:i + 1, ...].view(bs, -1, self.patch_size3, self.patch_size3)) \
                        .view(1, -1, self.patch_size3, self.patch_size3)
                )
            y_unfold = torch.cat(y_unfold, dim=0)
            x = F.fold(y_unfold.view(y_unfold.size(0), -1, bs).transpose(0, 2).contiguous(),
                       (self.patch_size1, self.patch_size1), kernel_size=self.patch_size3,
                       stride=self.patch_size_base)

            '''
            x_unfold = F.unfold(x, self.patch_size2, stride=self.patch_size_base).transpose(0, 2).contiguous()
            x_unfold = x_unfold.view(x_unfold.size(0), -1, self.patch_size2, self.patch_size2)
            y_unfold = []
            for i in range(x_unfold.size(0)):
                y_unfold.append(
                    self.stage4(x_unfold[i:i + 1, ...].view(bs, -1, self.patch_size2, self.patch_size2)) \
                        .view(1, -1, self.patch_size2, self.patch_size2)
                )
            y_unfold = torch.cat(y_unfold, dim=0)
            x = F.fold(y_unfold.view(y_unfold.size(0), -1, bs).transpose(0, 2).contiguous(),
                       (self.patch_size1, self.patch_size1), kernel_size=self.patch_size2,
                       stride=self.patch_size_base)

            x = self.stage5(x)
            '''
        elif self.args.flag == 3:
            bs, CH, w, h = x.size(0), x.size(1), x.size(2), x.size(3)  # batch_size

            x_unfold = F.unfold(x,
                                self.patch_size1,
                                stride=self.patch_size_base
                                ).view(-1, bs, CH, self.patch_size1, self.patch_size1).contiguous()
            y_unfold = []

            for i in range(x_unfold.size(0)):
                y_unfold.append(
                    self.stage1(x_unfold[i, ...])
                )

            y_unfold = torch.cat(y_unfold, dim=0).view(bs, self.patch_size1*self.patch_size1*CH, -1)

            x = F.fold(y_unfold, (w, h), kernel_size=self.patch_size1, stride=self.patch_size_base)

            # stage2
            x_unfold = F.unfold(x,
                                self.patch_size2,
                                stride=self.patch_size_base
                                ).view(-1, bs, CH, self.patch_size2, self.patch_size2).contiguous()
            y_unfold = []

            for i in range(x_unfold.size(0)):
                y_unfold.append(
                    self.stage2(x_unfold[i, ...])
                )

            y_unfold = torch.cat(y_unfold, dim=0).view(bs, self.patch_size2 * self.patch_size2 * CH, -1)

            x = F.fold(y_unfold, (w, h), kernel_size=self.patch_size2, stride=self.patch_size_base)
        elif self.args.flag == 4:
            bs, CH, w, h = x.size(0), x.size(1), x.size(2), x.size(3)  # batch_size

            x_unfold = F.unfold(x,
                                self.patch_size1,
                                stride=self.patch_size_base
                                ).permute(0, 2, 1).contiguous().view(-1, bs, CH, self.patch_size1, self.patch_size1)

            y_unfold = []

            for i in range(x_unfold.size(0)):
                y_unfold.append(
                    self.stage1(x_unfold[i, ...])
                )

            y_unfold = torch.cat(y_unfold, dim=0).view(bs, -1, self.patch_size1*self.patch_size1*CH).permute(0, 2, 1)

            x = F.fold(y_unfold, (w, h), kernel_size=self.patch_size1, stride=self.patch_size_base)

            # stage2
            x_unfold = F.unfold(x,
                                self.patch_size2,
                                stride=self.patch_size2
                                ).permute(0, 2, 1).contiguous().view(-1, bs, CH, self.patch_size2, self.patch_size2)
            y_unfold = []

            for i in range(x_unfold.size(0)):
                y_unfold.append(
                    self.stage2(x_unfold[i, ...])
                )

            y_unfold = torch.cat(y_unfold, dim=0).view(bs, -1, self.patch_size2 * self.patch_size2 * CH).permute(0, 2, 1)

            x = F.fold(y_unfold, (w, h), kernel_size=self.patch_size2, stride=self.patch_size2)

            # x = self.stage3(x)
        elif self.args.flag == 5:
            bs, CH, w, h = x.size(0), x.size(1), x.size(2), x.size(3)  # batch_size

            # stage2
            x_unfold = F.unfold(x,
                                self.patch_size2,
                                stride=self.patch_size2
                                ).permute(0, 2, 1).contiguous().view(-1, bs, CH, self.patch_size2, self.patch_size2)
            y_unfold = []

            for i in range(x_unfold.size(0)):
                y_unfold.append(
                    self.stage2(x_unfold[i, ...])
                )

            y_unfold = torch.cat(y_unfold, dim=0).view(bs, -1, self.patch_size2 * self.patch_size2 * CH).permute(0, 2,
                                                                                                                 1)

            x = F.fold(y_unfold, (w, h), kernel_size=self.patch_size2, stride=self.patch_size2)
        elif self.args.flag == 6:
            bs, CH, w, h = x.size(0), x.size(1), x.size(2), x.size(3)  # batch_size

            x = F.unfold(x, self.patch_size1,stride=self.patch_size_base).\
                permute(0, 2, 1).contiguous().view(-1, CH, self.patch_size1, self.patch_size1)
            # print("x_unfold", x_unfold.size())
            x = self.stage1(x)

            # print("y_unfold", y_unfold.size())
            x = x.view(bs, -1, self.patch_size1*self.patch_size1*CH).permute(0, 2, 1)
            x = F.fold(x, (w, h), kernel_size=self.patch_size1, stride=self.patch_size_base)

            # stage2
            x = F.unfold(x,
                                self.patch_size2,
                                stride=self.patch_size2
                                ).permute(0, 2, 1).contiguous().view(-1, CH, self.patch_size2, self.patch_size2)

            x = x.view(bs, -1, self.patch_size2 * self.patch_size2 * CH).permute(0, 2, 1)

            x = F.fold(x, (w, h), kernel_size=self.patch_size2, stride=self.patch_size2)

            # x = self.stage3(x)
        elif self.args.flag == 7:
            x = self.body(x)
        elif self.args.flag == 8:
            x = self.body(x)
        elif self.args.flag == 9:
            x = self.body1(x)
            x = self.body2(x)
            x = self.body3(x)
        elif self.args.flag == 10:
            x = self.body1(x)
            x = self.body2(x)
            x = self.body3(x)
            x = self.body4(x)
        elif self.args.flag == 11:

            x = self.body1(x)

            x = self.body2(x)

            x = self.body3(x)

            x = self.body4(x)

            x = self.body5(x)
        elif self.args.flag == 12:
            x = self.body(x)

        elif self.args.flag == 13:
            pass
        elif self.args.flag == 14:
            x = self.body(x)

        elif self.args.flag == 15:
            x = x + self.body1(x)

            x = x + self.body2(x)

        elif self.args.flag == 16:
            x = self.body(x)

        elif self.args.flag == 17:
            x1 = self.body1(x) + x

            x2 = self.body2(x1) + x1

            x3 = self.body3(x2) + x2

            x4 = self.body4(x3) + x3

            x = self.body5(x4)

        elif self.args.flag == 18:
            x1 = self.body1(x) + x

            x2 = self.body2(x1)

            x = self.body3(x2)

        elif self.args.flag == 19:
            x = self.body1(x)
            x = self.body2(x)
            x = self.body3(x)

        if self.args.flag == 20:
            x = self.body1(x)
            x = self.body2(x)

        out = self.tail(x)

        return y - out