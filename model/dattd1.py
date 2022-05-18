'''
    Dual agression and transform transformer-based Denoising(datt)
'''

import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from einops import rearrange
from option import args
import numpy as np
import numbers


def make_model(args, model_file_path=None):
    if args.mode == 'train':
        return DATTD(args), 1
    elif args.mode == 'val':
        model = DATTD(args)
        model.eval()
        return model, 1
    elif args.mode == 'test':
        if args.n_colors == 3:
            nb = 20
        else:
            nb = 17
        model = DATTD(args)
        # model = DnCNN(in_nc=args.n_colors, out_nc=args.n_colors, nc=64, nb=17, act_mode='BR')
        # model.load_state_dict(torch.load(args.model_file_name), strict=True)
        model.eval()

        return model, 1


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


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


class DATTD(nn.Module):
    def __init__(self, args, conv=default_conv):
        super(DATTD, self).__init__()

        self.scale_idx = 0

        self.args = args

        n_feats = args.n_feats
        kernel_size = 3
        act = nn.ReLU(True)

        num_heads = 16
        img_patch_size = args.patch_size
        window_size = args.patch_size // 2
        patch_dim = args.patch_dim

        self.sub_mean = MeanShift(args.rgb_range)
        self.add_mean = MeanShift(args.rgb_range, sign=1)

        self.head = nn.Sequential(
            conv(args.n_colors, n_feats, kernel_size),  # conv1
            ResBlock(conv, n_feats, kernel_size, act=act),  # conv2
        )

        # (32, 128, 128)
        self.body1 = Dual_block_1(n_feats, img_patch_size, num_head1=4, num_head2=4, win_size=args.patch_size // 8, patch_dim=2,
                                bias=True, LayerNorm_type='WithBias')
        self.body2 = Dual_block_1(n_feats, img_patch_size, num_head1=4, num_head2=4, win_size=args.patch_size // 8, patch_dim=2,
                                bias=True, LayerNorm_type='WithBias')
        self.fusion1 = CSATransformerBlock(n_feats * 2, num_heads=8, bias=True, LayerNorm_type='WithBias')
        self.patch_merge1 = Patch_Merge(n_feats*2, 4, 3)

        # (64, 96, 96)
        self.body3 = Dual_block_1(n_feats * 2, img_patch_size, num_head1=8, num_head2=4, win_size=args.patch_size // 8, patch_dim=2,
                                bias=True, LayerNorm_type='WithBias')
        self.body4 = Dual_block_1(n_feats * 2, img_patch_size, num_head1=8, num_head2=4, win_size=args.patch_size // 8, patch_dim=2,
                                bias=True, LayerNorm_type='WithBias')
        self.fusion2 = CSATransformerBlock(n_feats * 4, num_heads=8, bias=True, LayerNorm_type='WithBias')
        self.patch_merge2 = Patch_Merge(n_feats * 4, 3, 2)

        # (128, 64, 64)
        self.body5 = Dual_block_1(n_feats * 4, img_patch_size, num_head1=16, num_head2=4, win_size=args.patch_size // 4, patch_dim=2,
                                  bias=True, LayerNorm_type='WithBias')
        self.body6 = Dual_block_1(n_feats * 4, img_patch_size, num_head1=16, num_head2=4, win_size=args.patch_size // 4, patch_dim=2,
                                  bias=True, LayerNorm_type='WithBias')
        self.fusion3 = CSATransformerBlock(n_feats * 8, num_heads=8, bias=True, LayerNorm_type='WithBias')
        self.patch_merge3 = Patch_Merge(n_feats * 8, 2, 3)

        # (64, 96, 96)
        self.short_cut1 = nn.Conv2d(n_feats*8, n_feats*2, kernel_size=1)
        self.body7 = Dual_block_1(n_feats * 2, img_patch_size, num_head1=4, num_head2=4, win_size=args.patch_size // 8,
                                  patch_dim=2,
                                  bias=True, LayerNorm_type='WithBias')
        self.body8 = Dual_block_1(n_feats * 2, img_patch_size, num_head1=4, num_head2=4, win_size=args.patch_size // 8,
                                  patch_dim=2,
                                  bias=True, LayerNorm_type='WithBias')
        self.fusion4 = CSATransformerBlock(n_feats * 4, num_heads=8, bias=True, LayerNorm_type='WithBias')
        self.patch_merge4 = Patch_Merge(n_feats * 4, 3, 4)

        # (32, 128, 128)
        self.short_cut2 = nn.Conv2d(n_feats * 4, n_feats, kernel_size=1)
        self.body9 = Dual_block_1(n_feats, img_patch_size, num_head1=4, num_head2=4, win_size=args.patch_size // 8,
                                  patch_dim=2, bias=True, LayerNorm_type='WithBias')
        self.body10 = Dual_block_1(n_feats, img_patch_size, num_head1=4, num_head2=4, win_size=args.patch_size // 8,
                                  patch_dim=2, bias=True, LayerNorm_type='WithBias')
        self.fusion5 = CSATransformerBlock(n_feats * 2, num_heads=8, bias=True, LayerNorm_type='WithBias')

        self.tail = nn.Sequential(
            conv(n_feats * 2, args.n_colors, kernel_size)  # conv
        )

    def forward(self, x):
        y = x

        x = self.sub_mean(x)

        x = self.head(x)

        x1, x2 = self.body1(x, x)
        x1, x2 = self.body2(x1, x2)
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.fusion1(x1, x1, x1)
        o1 = self.patch_merge1(x1)

        x1, x2 = self.body3(o1, o1)
        x1, x2 = self.body4(x1, x2)
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.fusion2(x1, x1, x1)
        o2 = self.patch_merge1(x1)

        x1, x2 = self.body5(o2, o2)
        x1, x2 = self.body6(x1, x2)
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.fusion3(x1, x1, x1)
        o3 = self.patch_merge1(x1)

        o3 = self.short_cut1(torch.cat((o1, o3), dim=1))
        x1, x2 = self.body7(o3, o3)
        x1, x2 = self.body8(x1, x2)
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.fusion4(x1, x1, x1)
        o4 = self.patch_merge4(x1)

        o4 = self.short_cut2(torch.cat((o2, o4), dim=1))
        x1, x2 = self.body9(o4, o4)
        x1, x2 = self.body10(x1, x2)
        x1 = torch.cat((x1, x2), dim=1)
        x = self.fusion3(x1, x1, x1)

        x = self.tail(x)

        x = self.add_mean(x)

        return y - x


class Dual_block(nn.Module):
    ## double swin layer
    def __init__(self, n_feat, img_patch_size, num_head, win_size, patch_dim,
                 bias, LayerNorm_type,
                 ):
        super(Dual_block, self).__init__()

        self.swin_block = Cascade_SwinLayer(dim=n_feat, img_patch_size=img_patch_size, window_size=win_size,
                                            num_heads=num_head, patch_dim=patch_dim, bias=bias)
        self.CSA = CSATransformerBlock(dim=n_feat, num_heads=8, bias=bias, LayerNorm_type=LayerNorm_type)

    def forward(self, x1, x2):
        out1 = self.swin_block(x1, x2, x2)
        out2 = self.CSA(x2, x1, x1)

        return out1, out2


class Dual_block_1(nn.Module):
    ## single swin layer
    def __init__(self, n_feat, img_patch_size, num_head1, num_head2, win_size, patch_dim,
                 bias, LayerNorm_type,
                 ):
        super(Dual_block_1, self).__init__()

        self.swin_block = Singel_SwinLayer(dim=n_feat, img_patch_size=img_patch_size, window_size=win_size,
                                            num_heads=num_head1, patch_dim=patch_dim, bias=bias)
        self.CSA = CSATransformerBlock(dim=n_feat, num_heads=num_head2, bias=bias, LayerNorm_type=LayerNorm_type)

    def forward(self, x1, x2):
        out1 = self.swin_block(x1, x2, x2)
        out2 = self.CSA(x2, x1, x1)

        return out1, out2


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Multi_DConv_Head_Self_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Multi_DConv_Head_Self_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        # self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

    def forward(self, q, k, v):
        b, c, h, w = v.shape

        qkv = self.qkv_dwconv(torch.cat((self.q(q), self.k(k), self.v(v)), dim=1))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class CSATransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, bias, LayerNorm_type=' '):
        super(CSATransformerBlock, self).__init__()

        self.norm_q = LayerNorm(dim, LayerNorm_type)
        self.norm_k = LayerNorm(dim, LayerNorm_type)
        self.norm_v = LayerNorm(dim, LayerNorm_type)

        self.attn = Multi_DConv_Head_Self_Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)

        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=1)
        )

    def forward(self, q, k, v):
        v = v + self.attn(self.norm_q(q), self.norm_k(k), self.norm_v(v))
        v = v + self.ffn(v)
        return v


class Cascade_SwinLayer(nn.Module):
    def __init__(self, dim, img_patch_size, window_size, num_heads, patch_dim, bias=True):
        super(Cascade_SwinLayer, self).__init__()
        self.input_resolution = (img_patch_size // patch_dim, img_patch_size // patch_dim)
        self.win_resolution = window_size // patch_dim
        self.patch_dim = patch_dim

        self.patch_embed = Patch_Embed(1)
        self.patch_unembed = Patch_UnEmbed(1)

        self.q_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.qkv2 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)

        self.atten_proj_1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.atten_proj_2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.mlp1 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
                                  nn.ReLU())

        self.mlp2 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
                                  nn.ReLU())

        self.swinl1 = SwinTransformerBlock(dim * patch_dim * patch_dim, self.input_resolution, num_heads,
                                           window_size=self.win_resolution, shift_size=0)
        self.swinl2 = SwinTransformerBlock(dim * patch_dim * patch_dim, self.input_resolution, num_heads,
                                           window_size=self.win_resolution, shift_size=self.win_resolution // 2)

    def forward(self, q, k, v):
        shortcut = v
        q = self.patch_embed(Piexl_Shuffle_Invert(self.q_1(q), self.patch_dim))  # B, L, C
        k = self.patch_embed(Piexl_Shuffle_Invert(self.k_1(k), self.patch_dim))  # B, L, C
        v = self.patch_embed(Piexl_Shuffle_Invert(self.v_1(v), self.patch_dim))  # B, L, C

        v = shortcut + self.atten_proj_1(
            Piexl_Shuffle(self.patch_unembed(self.swinl1(q, k, v, self.input_resolution), self.input_resolution),
                          self.patch_dim))
        v = v + self.mlp1(v)

        qkv = self.qkv2(v)
        q, k, v = qkv.chunk(3, dim=1)
        shortcut = v
        q = self.patch_embed(Piexl_Shuffle_Invert(q, self.patch_dim))  # B, L, C
        k = self.patch_embed(Piexl_Shuffle_Invert(k, self.patch_dim))  # B, L, C
        v = self.patch_embed(Piexl_Shuffle_Invert(v, self.patch_dim))  # B, L, C

        v = shortcut + self.atten_proj_2(
            Piexl_Shuffle(self.patch_unembed(self.swinl2(q, k, v, self.input_resolution), self.input_resolution),
                          self.patch_dim))
        v = v + self.mlp2(v)

        return v


class Singel_SwinLayer(nn.Module):
    def __init__(self, dim, img_patch_size, window_size, num_heads, patch_dim, bias=True):
        super(Singel_SwinLayer, self).__init__()

        self.input_resolution = (img_patch_size // patch_dim, img_patch_size // patch_dim)
        self.win_resolution = window_size // patch_dim
        self.patch_dim = patch_dim

        self.patch_embed = Patch_Embed(1)
        self.patch_unembed = Patch_UnEmbed(1)

        self.q_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.atten_proj_1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.mlp1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.swin = SwinTransformerBlock(dim * patch_dim * patch_dim, self.input_resolution, num_heads,
                                         window_size=self.win_resolution, shift_size=0)

    def forward(self, q, k, v):
        shortcut = v
        q = self.patch_embed(Piexl_Shuffle_Invert(self.q_1(q), self.patch_dim))  # B, L, C
        k = self.patch_embed(Piexl_Shuffle_Invert(self.k_1(k), self.patch_dim))  # B, L, C
        v = self.patch_embed(Piexl_Shuffle_Invert(self.v_1(v), self.patch_dim))  # B, L, C

        v = shortcut + self.atten_proj_1(
            Piexl_Shuffle(
                self.patch_unembed(
                    self.swin(q, k, v, self.input_resolution), self.input_resolution), self.patch_dim))

        v = v + self.mlp1(v)

        return v


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = v.shape
        qkv = torch.cat((q, k, v), dim=1).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        # attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        # x = self.proj(x)
        # x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        # self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        # self.norm1 = norm_layer(dim)
        self.norm_q = norm_layer(dim)
        self.norm_k = norm_layer(dim)
        self.norm_v = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        # mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, q, k, v, x_size):
        H, W = x_size
        B, L, C = v.shape
        # assert L == H * W, "input feature has wrong size"

        # print(q.shape)
        shortcut = v
        q = self.norm_q(q).view(B, H, W, C)
        k = self.norm_k(k).view(B, H, W, C)
        v = self.norm_v(v).view(B, H, W, C)
        # cyclic shift
        if self.shift_size > 0:
            shifted_q = torch.roll(q, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_k = torch.roll(k, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_v = torch.roll(v, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_q = q
            shifted_k = k
            shifted_v = v

        # partition windows
        q_windows = window_partition(shifted_q, self.window_size)  # nW*B, window_size, window_size, C
        q_windows = q_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        k_windows = window_partition(shifted_k, self.window_size)  # nW*B, window_size, window_size, C
        k_windows = k_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        v_windows = window_partition(shifted_v, self.window_size)  # nW*B, window_size, window_size, C
        v_windows = v_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(q_windows, k_windows, v_windows,
                                     mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(q_windows, k_windows, v_windows, mask=self.calculate_mask(x_size).to(v.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        x = x + shortcut

        '''
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        '''

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Patch_Embed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        patch_size:
    Return:
        x: (B, L, d_model), L:Num patches, d_model:patch_dim*patch_dim*C
    """

    def __init__(self, patch_dim=3):
        super().__init__()
        self.patch_dim = patch_dim

    def forward(self, x):
        x = F.unfold(x, self.patch_dim, stride=self.patch_dim).transpose(1, 2).contiguous()  # shape == (B, L, d_model)

        return x


class Patch_UnEmbed(nn.Module):
    r""" Image to Patch Embedding
    __init__():
        Args:
            patch_size
    forward():
        Arges:
            x: input
            out_size(): a tupele, the shape of out_size
        return:
            x: shape:(B, C, out_size[0], out_size[1])
    """

    def __init__(self, patch_dim=3):
        super().__init__()
        self.patch_dim = patch_dim

    def forward(self, x, out_size):
        x = F.fold(x.transpose(1, 2).contiguous(),
                   output_size=out_size,
                   kernel_size=(self.patch_dim, self.patch_dim),
                   stride=self.patch_dim)  # B, C, W, H
        return x


def Piexl_Shuffle_Invert(x, down_factor):
    r""" Image downsample
    Args:
        x: B C H W
        down_factor(int): The factor of downsampling, generally equal to patchsize
    return:
        shape: B C*down_factor*down_factor H/down_factor W/down_factor
    """
    B, C, W, H = x.shape
    assert W % down_factor == 0
    assert H % down_factor == 0

    return F.unfold(x, down_factor, stride=down_factor).view(B, -1, int(W / down_factor),
                                                             int(H / down_factor)).contiguous()


def Piexl_Shuffle(x, up_factor):
    r""" Image upsample
        Args:
            x: B C H W
            up_factor(int): The factor of upsampling, generally equal to patchsize
        return:
            shape: B C/up_factor/up_factor H*up_factor W*up_factor
    """
    pixel_shuffle = nn.PixelShuffle(up_factor)

    return pixel_shuffle(x)


class Patch_Merge(nn.Module):
    r""" Image to Patch Embedding
    __init__():
        Args:
            patch_size
    forward():
        Arges:
            x: input
            out_size(): a tupele, the shape of out_size
        return:
            x: shape:(B, C, out_size[0], out_size[1])
    """

    def __init__(self, dim, in_factor, out_factor):
        super().__init__()
        self.in_factor = in_factor
        self.out_factor = out_factor
        self.dim =dim

        self.trans = nn.Conv2d(dim*in_factor*in_factor, dim*out_factor*out_factor, kernel_size=1)

    def forward(self, x, out_size):
        x = Piexl_Shuffle_Invert(x, self.in_factor)
        x = self.trans(x)
        x = Piexl_Shuffle(x, up_factor=self.out_factor)

        return x


class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0, 1.0), sign=-1
    ):
        super(MeanShift, self).__init__(4, 4, kernel_size=(1, 1))
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(4).view(4, 4, 1, 1) / std.view(4, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


if __name__ == "__main__":
    args.patch_size = 96
    args.n_colors = 4
    test_input = torch.from_numpy(np.random.randn(1, 4, 96, 96)).float().cuda()
    # net = Unet().cuda()
    net = DATTD(args).cuda()
    output = net(test_input)
    print("test over")
    # import time
    # time.sleep(1000)
    print(str(torch.cuda.is_available()))