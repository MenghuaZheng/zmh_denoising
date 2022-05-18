'''
    Window based transformer Layer——WBTransformerLayer
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from model_common.common import default_conv as conv


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

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
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

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
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

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, shift_dir='',
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        if shift_dir == 'no_shift':
            self.shift_size = 0

        self.shift_dir = shift_dir
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution, k=self.shift_dir)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size, k):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        if k == 'vertical':
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, None))
        elif k == 'horizontal':
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
        else:
            raise ValueError("shift_dir equal vertical or horizontal")

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

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            if self.shift_dir == 'vertical':
                shifted_x = torch.roll(x, shifts=(-self.shift_size, 0), dims=(1, 2))
            elif self.shift_dir == 'horizontal':
                shifted_x = torch.roll(x, shifts=(0, -self.shift_size), dims=(1, 2))
            else:
                raise ValueError("shift_dir equal vertical or horizontal")
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            if self.shift_dir == 'vertical':
                x = torch.roll(shifted_x, shifts=(self.shift_size, 0), dims=(1, 2))
            elif self.shift_dir == 'horizontal':
                x = torch.roll(shifted_x, shifts=(0, self.shift_size), dims=(1, 2))
            else:
                raise ValueError("shift_dir equal vertical or horizontal")
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

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


class WBformerLayer(nn.Module):
    def __init__(self, dim,
                 img_size,
                 num_heads,
                 patch_dim,
                 window_size,
                 layers,
                 drop_path,
                 mlp_ratio=4.,
                 ape=False,
                 ):
        super().__init__()
        assert img_size % patch_dim == 0
        assert window_size % patch_dim == 0

        self.patch_dim = patch_dim

        self.patch_embed = Patch_Embed(1)
        self.patch_unembed = Patch_UnEmbed(1)
        self.ape = ape

        self.input_resolution = (int(img_size/patch_dim), int(img_size/patch_dim))
        window_size = int(window_size / patch_dim)
        shift_size = int(window_size / 2)
        embed_dim = dim*patch_dim*patch_dim

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1,
                                                               self.input_resolution[0]*self.input_resolution[1],
                                                               embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.layers = nn.ModuleList()
        for i_layer in range(len(layers)):
            self.layer = SwinTransformerBlock(dim=embed_dim,
                                               input_resolution=self.input_resolution,
                                               num_heads=num_heads,
                                               window_size=window_size,
                                               shift_size=shift_size,
                                               shift_dir=layers[i_layer],
                                               mlp_ratio=mlp_ratio,
                                               drop_path=drop_path[i_layer])
            self.layers.append(self.layer)

    def forward(self, x):
        x = self.patch_embed(Piexl_Shuffle_Invert(x, self.patch_dim))   # B, L, C

        if self.ape:
            x = x + self.absolute_pos_embed

        for layer in self.layers:
            x = layer(x, self.input_resolution)

        x = Piexl_Shuffle(self.patch_unembed(x, self.input_resolution), self.patch_dim)

        return x


class CBformerLayer(nn.Module):
    def __init__(self, dim,
                 img_size,
                 num_heads,
                 patch_dim,
                 kernel_size=5,
                 ape=False,
                 dropout=0.1,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm
                 ):
        super().__init__()
        assert img_size % patch_dim == 0

        self.patch_dim = patch_dim

        self.patch_embed = Patch_Embed(1)
        self.patch_unembed = Patch_UnEmbed(1)
        self.ape = ape

        self.input_resolution = (int(img_size/patch_dim), int(img_size/patch_dim))
        embed_dim = dim*patch_dim*patch_dim

        self.norm1 = norm_layer(embed_dim)
        self.norm2 = norm_layer(embed_dim)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, self.input_resolution[0]*self.input_resolution[1], embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, bias=False)

        self.conv = conv(dim, dim, kernel_size)
        # self.conv1 = conv(dim, dim, kernel_size)
        self.act = act_layer()
        # self.conv2 = conv(dim, dim, kernel_size)

    def forward(self, x):
        x = self.patch_embed(Piexl_Shuffle_Invert(x, self.patch_dim))   # B, L, C

        short_cut = x
        x = self.norm1(x)

        if self.ape:
            x = x + self.absolute_pos_embed

        x = x.transpose(0, 1).contiguous()

        x, _ = self.self_attn(x, x, x)

        x = x.transpose(0, 1).contiguous()

        x = Piexl_Shuffle(self.patch_unembed(self.norm2(x + short_cut), self.input_resolution), self.patch_dim)

        # x = x + self.conv2(self.act(self.conv1(x)))

        x = x + self.act(self.conv(x))

        return x


class CBformerLayer1(nn.Module):
    def __init__(self, dim,
                 img_size,
                 win_size,
                 num_heads,
                 patch_dim,
                 kernel_size=5,
                 ape=False,
                 dropout=0.1,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm
                 ):
        super().__init__()
        assert win_size % patch_dim == 0

        self.patch_dim = patch_dim

        self.patch_embed = Patch_Embed(1)
        self.patch_unembed = Patch_UnEmbed(1)
        self.ape = ape

        self.input_resolution = (int(win_size/patch_dim), int(win_size/patch_dim))

        self.img_size = (img_size, img_size)

        self.win_size = win_size

        if img_size <= win_size:
            self.win_embed = None
        else:
            self.win_embed = self.window_partition
            self.win_embed_reserve = self.window_partition_reserve

        embed_dim = dim*patch_dim*patch_dim

        self.norm1 = norm_layer(embed_dim)
        self.norm2 = norm_layer(embed_dim)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, self.input_resolution[0]*self.input_resolution[1], embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, bias=False)

        self.conv = conv(dim, dim, kernel_size)
        # self.conv1 = conv(dim, dim, kernel_size)
        self.act = act_layer()
        # self.conv2 = conv(dim, dim, kernel_size)

    def forward(self, x):
        if self.win_embed is not None:
            x = self.win_embed(x, win_size=self.win_size)

        x = self.patch_embed(Piexl_Shuffle_Invert(x, self.patch_dim))   # B, L, C

        short_cut = x
        x = self.norm1(x)

        if self.ape:
            x = x + self.absolute_pos_embed

        x = x.transpose(0, 1).contiguous()

        x, _ = self.self_attn(x, x, x)

        x = x.transpose(0, 1).contiguous()

        x = Piexl_Shuffle(self.patch_unembed(self.norm2(x + short_cut), self.input_resolution), self.patch_dim)

        # x = x + self.conv2(self.act(self.conv1(x)))

        x = x + self.act(self.conv(x))

        if self.win_embed is not None:
            x = self.win_embed_reserve(x, output_size=self.img_size, win_size=self.win_size)

        return x

    def window_partition(self, input, win_size):
        _, C, _, _ = input.shape

        # print(input.shape)
        return F.unfold(input, win_size, stride=win_size).permute(0, 2, 1).contiguous().view(-1, C, win_size, win_size)  # shape == (B*n_win, C, win_size, win_size)

    def window_partition_reserve(self, input, output_size, win_size):
        _, C, _, _ = input.shape

        return F.fold(input.view(1, -1, C * win_size * win_size*4).permute(0, 2, 1).contiguous(),
                      output_size=output_size,
                      kernel_size=(win_size, win_size),
                      stride=win_size)


class Dual_WBformaerlayer(nn.Module):
    def __init__(self,
                 dim,
                 img_size,
                 layers1,
                 layers2,
                 num_heads1,
                 num_heads2,
                 patch_dim1,
                 patch_dim2,
                 drop_path1=[0., 0., 0.],
                 drop_path2=[0., 0., 0.],
                 mlp_ratio=4.,
                 ape=False,
                 kernel_size=1,
                 act_layer=nn.GELU
                 ):
        super().__init__()

        self.n_feat = dim

        self.window_size1 = int(img_size/2)
        self.window_size2 = int(img_size/3)

        self.body1 = WBformerLayer(dim=self.n_feat,
                                   img_size=img_size,
                                   num_heads=num_heads1,
                                   patch_dim=patch_dim1,
                                   window_size=self.window_size1,
                                   layers=layers1,
                                   drop_path=drop_path1,
                                   mlp_ratio=mlp_ratio,
                                   ape=ape)

        self.body2 = WBformerLayer(dim=self.n_feat,
                                   img_size=img_size,
                                   num_heads=num_heads2,
                                   patch_dim=patch_dim2,
                                   window_size=self.window_size2,
                                   layers=layers2,
                                   drop_path=drop_path2,
                                   mlp_ratio=mlp_ratio,
                                   ape=ape)

        self.conv1 = conv(dim*2, dim*4, kernel_size)
        self.conv2 = conv(dim*4, dim, kernel_size)
        self.act = act_layer()

    def forward(self, x):
        x = torch.cat((self.body1(x), self.body2(x)), dim=1)

        x = self.conv2(self.act(self.conv1(x)))

        return x


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
                   stride=self.patch_dim)    # B, C, W, H
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

    return F.unfold(x, down_factor, stride=down_factor).view(B, -1, int(W/down_factor), int(H/down_factor)).contiguous()


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


