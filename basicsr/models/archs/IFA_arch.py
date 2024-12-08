import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import numbers
from einops import rearrange

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

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
        return x / torch.sqrt(sigma+1e-5) * self.weight

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
        mu = x.mean(dim=-1, keepdim=True)
        sigma = x.var(dim=-1, keepdim=True)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm1(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm1, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
class FeedForward1(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward1, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features , dim, kernel_size=1, bias=bias)

    def forward(self, x):
        #print(f"Input shape: {x.shape}")
        x = self.project_in(x)
        #print(f"After project_in: {x.shape}")
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        #print(f"After dwconv: {x1.shape}, {x2.shape}")
        x = F.gelu(x1) * x2
        #print(f"After GELU and multiplication: {x.shape}")
        x = self.project_out(x)
        #print(f"After project_out: {x.shape}")
        return x
class Attention1(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention1, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        kv = self.kv_dwconv(self.kv(x))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(y))

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

class IFA(nn.Module):
    def __init__(self, dim_21=40, dim1=48, num_heads=2, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
        super(IFA, self).__init__()

        self.conv1 = nn.Conv2d(dim_21, dim1, (1, 1))
        # self.conv2 = nn.Conv2d(dim, dim_2, (1, 1))
        self.norm1 = LayerNorm1(dim1, LayerNorm_type)
        self.attn = Attention1(dim1, num_heads, bias)
        self.norm2 = LayerNorm1(dim1, LayerNorm_type)
        self.ffn = FeedForward1(dim1, ffn_expansion_factor, bias)

    def forward(self, input_R, input_S):
        # input_ch = input_R.size()[1]
        input_S = F.interpolate(input_S, [input_R.shape[2], input_R.shape[3]])
        input_S = self.conv1(input_S)
        # input_S = F.interpolate(input_S, size=input_size, mode='bilinear', align_corners=True)
        input_R = self.norm1(input_R)
        input_S = self.norm1(input_S)
        input_R = input_R + self.attn(input_R, input_S)
        input_R = input_R + self.ffn(self.norm2(input_R))
        return input_R

class IFA1(nn.Module):
    def __init__(self, dim_21=40, dim1=704, num_heads=2, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
        super(IFA1, self).__init__()

        self.conv1 = nn.Conv2d(dim_21, dim1, (1, 1))
        # self.conv2 = nn.Conv2d(dim, dim_2, (1, 1))
        self.norm1 = LayerNorm1(dim1, LayerNorm_type)
        self.attn = Attention1(dim1, num_heads, bias)
        self.norm2 = LayerNorm1(dim1, LayerNorm_type)
        self.ffn = FeedForward1(dim1, ffn_expansion_factor, bias)

    def forward(self, input_R, input_S):
        # input_ch = input_R.size()[1]
        input_S = F.interpolate(input_S, [input_R.shape[2], input_R.shape[3]])
        input_S = self.conv1(input_S)
        # input_S = F.interpolate(input_S, size=input_size, mode='bilinear', align_corners=True)
        input_R = self.norm1(input_R)
        input_S = self.norm1(input_S)
        input_R = input_R + self.attn(input_R, input_S)
        input_R = input_R + self.ffn(self.norm2(input_R))
        return input_R

class IFA2(nn.Module):
    def __init__(self, dim_222=40, dim22=320, num_heads=2, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
        super(IFA2, self).__init__()

        self.conv1 = nn.Conv2d(dim_222, dim22, (1, 1))
        # self.conv2 = nn.Conv2d(dim, dim_2, (1, 1))
        self.norm1 = LayerNorm1(dim22, LayerNorm_type)
        self.attn = Attention1(dim22, num_heads, bias)
        self.norm2 = LayerNorm1(dim22, LayerNorm_type)
        self.ffn = FeedForward1(dim22, ffn_expansion_factor, bias)

    def forward(self, input_R2, input_S2):
        # input_ch = input_R.size()[1]
        input_S2 = F.interpolate(input_S2, [input_R2.shape[2], input_R2.shape[3]])
        input_S2 = self.conv1(input_S2)
        # input_S = F.interpolate(input_S, size=input_size, mode='bilinear', align_corners=True)
        input_R2 = self.norm1(input_R2)
        input_S2 = self.norm1(input_S2)
        input_R2 = input_R2 + self.attn(input_R2, input_S2)
        input_R2 = input_R2 + self.ffn(self.norm2(input_R2))
        return input_R2

class IFA3(nn.Module):
    def __init__(self, dim_23=40, dim3=160, num_heads=2, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
        super(IFA3, self).__init__()

        self.conv1 = nn.Conv2d(dim_23, dim3, (1, 1))
        # self.conv2 = nn.Conv2d(dim, dim_2, (1, 1))
        self.norm1 = LayerNorm1(dim3, LayerNorm_type)
        self.attn = Attention1(dim3, num_heads, bias)
        self.norm2 = LayerNorm1(dim3, LayerNorm_type)
        self.ffn = FeedForward1(dim3, ffn_expansion_factor, bias)

    def forward(self, input_R, input_S):
        # input_ch = input_R.size()[1]
        input_S = F.interpolate(input_S, [input_R.shape[2], input_R.shape[3]])
        input_S = self.conv1(input_S)
        # input_S = F.interpolate(input_S, size=input_size, mode='bilinear', align_corners=True)
        input_R = self.norm1(input_R)
        input_S = self.norm1(input_S)
        input_R = input_R + self.attn(input_R, input_S)
        input_R = input_R + self.ffn(self.norm2(input_R))
        return input_R

#821 有用
# class IFA(nn.Module):
#     def __init__(self, num_heads=2, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
#         super(IFA, self).__init__()
#         self.num_heads = num_heads
#         self.ffn_expansion_factor = ffn_expansion_factor
#         self.bias = bias
#         self.LayerNorm_type = LayerNorm_type
#
#         self.conv1 = None  # 初始化为None，稍后动态创建
#         self.norm1 = None  # 初始化为None，稍后动态创建
#         self.attn = None   # 初始化为None，稍后动态创建
#         self.norm2 = None  # 初始化为None，稍后动态创建
#         self.ffn = None    # 初始化为None，稍后动态创建
#
#     def forward(self, input_R, input_S):
#         # 动态获取dim和dim_2的值
#         dim = input_R.size(1)
#         dim_2 = input_S.size(1)
#
#         # 动态创建卷积层、归一化层、注意力模块和前馈网络
#         if self.conv1 is None or self.conv1.in_channels != dim_2 or self.conv1.out_channels != dim:
#             self.conv1 = nn.Conv2d(dim_2, dim, (1, 1)).to(input_R.device)
#             self.norm1 = LayerNorm1(dim, self.LayerNorm_type).to(input_R.device)
#             self.attn = Attention1(dim, self.num_heads, self.bias).to(input_R.device)
#             self.norm2 = LayerNorm1(dim, self.LayerNorm_type).to(input_R.device)
#             self.ffn = FeedForward1(dim, self.ffn_expansion_factor, self.bias).to(input_R.device)
#
#         # 调整input_S的尺寸并通过conv1
#         input_S = F.interpolate(input_S, [input_R.shape[2], input_R.shape[3]])
#         input_S = self.conv1(input_S)
#
#         # 对input_R和input_S进行归一化
#         input_R = self.norm1(input_R)
#         input_S = self.norm1(input_S)
#
#         # 计算注意力和前馈网络的结果
#         input_R = input_R + self.attn(input_R, input_S)
#         input_R = input_R + self.ffn(self.norm2(input_R))
#
#         return input_R


#
#
# ifa = IFA2(num_heads=2, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias')
#
# input_R2 = torch.randn(6, 192, 32, 32)
# input_S2 = torch.randn(6, 40, 128, 128)
# #
# output = ifa(input_R2, input_S2)
# print(f'xxde1{output.shape}')


# illu_fea shape: torch.Size([1, 40, 400, 600])
# latent after concatenation shape: torch.Size([1, 704, 50, 75])

# if __name__ == '__main__':
#     # 假设输入的通道数 c = 64, 图像尺寸 h = w = 32
#     input_channels = 40
#     image_height = 256
#     image_width = 256
#     batch_size = 1  # 假设批大小为4
#
#     # 创建IFA的实例
#     # 假设我们使用以下参数：
#     dim = 40  # 输出通道数
#     num_heads = 8 # 注意力头数
#     ffn_expansion_factor = 2.0  # 前馈网络的维度扩展因子
#     bias = True  # 是否在卷积中使用偏置
#     LayerNorm_type = 'WithBias'  # 使用带偏置的层归一化
#
#     # 实例化IFA
#     transformer_block = IFA(dim_2=input_channels, dim=dim, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
#
#     # 创建一个假的输入张量以测试模型
#     input_R = torch.randn(batch_size, input_channels, image_height, image_width)
#     input_S = torch.randn(batch_size, input_channels, image_height, image_width)
#
#     # 执行前向传播
#     output = transformer_block(input_R, input_S)
#     print(output.shape)  # 应该输出形状为 [b, c, h, w] , torch.Size([1, 40, 256, 256])