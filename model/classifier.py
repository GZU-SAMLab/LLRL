import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from model.utils import weight_init

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """随机丢弃路径（Stochastic Depth），在残差块的主路径中每个样本独立应用。"""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # 适用于不同维度的张量，而不仅仅是2D卷积网络
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # 二值化
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """随机丢弃路径的模块化实现"""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Mlp(nn.Module):
    """MLP，作为Vision Transformer、MLP-Mixer及相关网络中的组件"""
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

class CrossAttention(nn.Module):
    """跨注意力模块"""
    def __init__(self, dim1, dim2, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim1 // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim1, dim1, bias=qkv_bias)
        self.kv = nn.Linear(dim2, dim1 * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim1, dim1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        B1, N1, C1 = x.shape
        B2, N2, C2 = y.shape

        # 计算查询向量
        q = self.q(x).reshape(B1, N1, self.num_heads, C1 // self.num_heads).permute(0, 2, 1, 3)
        # 计算键和值向量
        kv = self.kv(y).reshape(B2, N2, 2, self.num_heads, C1 // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        # 计算注意力得分
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 计算注意力加权输出
        x = (attn @ v).transpose(1, 2).reshape(B1, N1, C1)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    """基本块，包括归一化、跨注意力和MLP"""
    def __init__(self, dim1, dim2, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim1)
        self.norm2 = norm_layer(dim2)
        self.attn = CrossAttention(dim1, dim2, num_heads, qkv_bias, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm3 = norm_layer(dim1)
        mlp_hidden_dim = int(dim1 * mlp_ratio)
        self.mlp = Mlp(in_features=dim1, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, y):
        # 残差连接和跨注意力
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm2(y)))
        # 残差连接和MLP
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x


class FeatureInjector(nn.Module):
    """特征注入模块"""
    def __init__(self, dim1=384, dim2=[64, 128, 256], num_heads=8, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                    drop_path=0., act_layer=nn.ReLU, norm_layer=nn.LayerNorm):
        super().__init__()

        # 不同层之间的跨注意力模块
        self.c2_c5 = Block(dim1, dim2[0], num_heads, mlp_ratio, qkv_bias, drop, attn_drop, drop_path, act_layer, norm_layer)
        self.c3_c5 = Block(dim1, dim2[1], num_heads, mlp_ratio, qkv_bias, drop, attn_drop, drop_path, act_layer, norm_layer)
        self.c4_c5 = Block(dim1, dim2[2], num_heads, mlp_ratio, qkv_bias, drop, attn_drop, drop_path, act_layer, norm_layer)

        # 融合层
        self.fuse = nn.Conv2d(dim1*3, dim1, 1, bias=False)

        weight_init(self)


    def base_forward(self, c2, c3, c4, c5):
        H, W = c5.shape[2:]

        # 调整张量形状
        c2 = rearrange(c2, 'b c h w -> b (h w) c')
        c3 = rearrange(c3, 'b c h w -> b (h w) c')
        c4 = rearrange(c4, 'b c h w -> b (h w) c')
        c5 = rearrange(c5, 'b c h w -> b (h w) c')

        # 通过跨注意力模块
        _c2 = self.c2_c5(c5, c2)
        _c2 = rearrange(_c2, 'b (h w) c -> b c h w', h=H, w=W)

        _c3 = self.c3_c5(c5, c3)
        _c3 = rearrange(_c3, 'b (h w) c -> b c h w', h=H, w=W)

        _c4 = self.c4_c5(c5, c4)
        _c4 = rearrange(_c4, 'b (h w) c -> b c h w', h=H, w=W)

        # 融合不同层的特征
        _c5 = self.fuse(torch.cat([_c2, _c3, _c4], dim=1))

        return _c5

    def forward(self, fx, fy):
        # 计算两个输入特征的融合结果
        _c5x = self.base_forward(fx[0], fx[1], fx[2], fx[3])
        _c5y = self.base_forward(fy[0], fy[1], fy[2], fy[3])

        return _c5x, _c5y

# 消融实验特征增强
class FeatureInjector1(nn.Module):
    """特征注入模块"""
    def __init__(self, dim1=384, dim2=[64, 128, 256], num_heads=8, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                    drop_path=0., act_layer=nn.ReLU, norm_layer=nn.LayerNorm):
        super().__init__()

        # 不同层之间的跨注意力模块
        self.c2_c5 = Block(dim1, dim2[0], num_heads, mlp_ratio, qkv_bias, drop, attn_drop, drop_path, act_layer, norm_layer)
        self.c3_c5 = Block(dim1, dim2[1], num_heads, mlp_ratio, qkv_bias, drop, attn_drop, drop_path, act_layer, norm_layer)
        self.c4_c5 = Block(dim1, dim2[2], num_heads, mlp_ratio, qkv_bias, drop, attn_drop, drop_path, act_layer, norm_layer)

        # 融合层
        self.fuse = nn.Conv2d(dim1*3, dim1, 1, bias=False)

        weight_init(self)


    def base_forward(self, c2, c3, c4, c5):
        H, W = c5.shape[2:]

        # 调整张量形状
        c2 = rearrange(c2, 'b c h w -> b (h w) c')
        c3 = rearrange(c3, 'b c h w -> b (h w) c')
        c4 = rearrange(c4, 'b c h w -> b (h w) c')
        c5 = rearrange(c5, 'b c h w -> b (h w) c')

        # 通过跨注意力模块
        _c2 = self.c2_c5(c2, c5)
        _c2 = rearrange(_c2, 'b (h w) c -> b c h w', h=H, w=W)

        _c3 = self.c3_c5(c3, c5)
        _c3 = rearrange(_c3, 'b (h w) c -> b c h w', h=H, w=W)

        _c4 = self.c4_c5(c4, c5)
        _c4 = rearrange(_c4, 'b (h w) c -> b c h w', h=H, w=W)

        # 融合不同层的特征
        _c5 = self.fuse(torch.cat([_c2, _c3, _c4], dim=1))

        return _c5

    def forward(self, fx, fy):
        # 计算两个输入特征的融合结果
        _c5x = self.base_forward(fx[0], fx[1], fx[2], fx[3])
        _c5y = self.base_forward(fy[0], fy[1], fy[2], fy[3])

        return _c5x, _c5y


class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class SelfAttention(nn.Module):
    """自注意力模块"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, C, H, W = q.shape

        # 将输入展平
        q = q.view(B, C, H * W).permute(0, 2, 1)  # B, N, C
        k = k.view(B, C, H * W).permute(0, 2, 1)  # B, N, C
        v = v.view(B, C, H * W).permute(0, 2, 1)  # B, N, C

        # 计算注意力得分
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 计算注意力加权输出
        x = (attn @ v).transpose(1, 2).reshape(B, -1, self.dim)
        x = x.view(B, H * W, C).permute(0, 2, 1).view(B, C, H, W)

        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, C, H, W = q.shape

        # 将输入展平
        q = q.view(B, C, H * W).permute(0, 2, 1)  # B, N, C
        k = k.view(B, C, H * W).permute(0, 2, 1)  # B, N, C
        v = v.view(B, C, H * W).permute(0, 2, 1)  # B, N, C

        # 线性变换
        q = self.q_proj(q)  # B, N, C
        k = self.k_proj(k)  # B, N, C
        v = self.v_proj(v)  # B, N, C

        # 多头分割
        q = q.view(B, -1, self.num_heads, self.dim // self.num_heads).permute(0, 2, 1, 3)  # B, num_heads, N, head_dim
        k = k.view(B, -1, self.num_heads, self.dim // self.num_heads).permute(0, 2, 1, 3)  # B, num_heads, N, head_dim
        v = v.view(B, -1, self.num_heads, self.dim // self.num_heads).permute(0, 2, 1, 3)  # B, num_heads, N, head_dim

        # 计算注意力得分
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 计算注意力加权输出
        x = (attn @ v).permute(0, 2, 1, 3).contiguous().view(B, -1, self.dim)  # B, N, C
        x = self.proj(x)
        x = self.proj_drop(x)

        # 恢复原始维度
        x = x.view(B, H * W, C).permute(0, 2, 1).view(B, C, H, W)

        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),  # 512 -> 32
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)  # 32 -> 512
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return self.sigmoid(out) * x

class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)  # Apply channel attention
        x = self.spatial_attention(x)  # Apply spatial attention
        return x

class Classifier(nn.Module):
    """增强的分类器模块"""
    def __init__(self, in_dim=[64, 128, 256, 512], num_class=6, decay=4):
        super().__init__()
        c2_channel, c3_channel, c4_channel, c5_channel = in_dim

        self.structure_enhance = FeatureInjector(dim1=c5_channel)

        self.down_c2 = nn.Sequential(
            nn.Conv2d(c2_channel, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.down_c3 = nn.Sequential(
            nn.Conv2d(c3_channel, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.down_c4 = nn.Sequential(
            nn.Conv2d(c4_channel, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # 自注意力机制
        self.self_attention = SelfAttention(dim=512, num_heads=2)

        #
        self.AttentionModule = AttentionModule(in_channels=512)

        # 定义多个MLP模块，每个MLP处理不同层的特征
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim * 3, dim // decay, 1, bias=False),
                nn.BatchNorm2d(dim // decay),
                nn.ReLU(),
                nn.Conv2d(dim // decay, dim // decay, 3, 1, padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(dim // decay, dim // decay, 3, 1, padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(dim // decay, dim, 3, 1, padding=1, bias=False)
            ) for dim in in_dim
        ])

        # 增强的分类器头部
        self.conv1 = nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(1024)
        self.relu = nn.ReLU(inplace=True)
        self.res_block = ResidualBlock(1024, 512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_class)

    def forward(self, fx, fy):
        c2x, c3x, c4x, c5x = fx

        c5x, c5y = self.structure_enhance(fx, fy)

        # 下采样到目标大小 [16, 384, 14, 14]
        c2f = self.down_c2(c2x)
        c3f = self.down_c3(c3x)
        c4f = self.down_c4(c4x)
        # print(f'Initial shapes - c5x: {c5x.shape}, c4x: {c4f.shape}, c3x: {c3f.shape}, c2x: {c2f.shape}')

        # # 自注意力机制
        # attn_output = self.self_attention(c4f, c3f, c2f)
        # # 自注意力消融实验
        # # attn_output = self.self_attention(c2f, c3f, c4f)
        # # 将注意力特征与增强后的特征进行融合
        # fused_feature = attn_output + c5x

        # 通道注意力+空间注意力
        fused_feature = c2f + c3f + c4f + c5x
        fused_feature = self.AttentionModule(fused_feature)

        # 增强分类器头部的前向传播
        c5x = self.conv1(fused_feature)
        c5x = self.bn1(c5x)
        c5x = self.relu(c5x)
        c5x = self.res_block(c5x)
        c5x = self.avgpool(c5x)
        c5x = torch.flatten(c5x, 1)
        pred_x = self.fc(c5x)

        return pred_x

class ClassifierFeatureVision(nn.Module):
    """增强的分类器模块"""
    def __init__(self, in_dim=[64, 128, 256, 512], num_class=6, decay=4):
        super().__init__()
        c2_channel, c3_channel, c4_channel, c5_channel = in_dim

        self.structure_enhance = FeatureInjector(dim1=c5_channel)

        self.down_c2 = nn.Sequential(
            nn.Conv2d(c2_channel, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.down_c3 = nn.Sequential(
            nn.Conv2d(c3_channel, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.down_c4 = nn.Sequential(
            nn.Conv2d(c4_channel, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # 自注意力机制
        self.self_attention = SelfAttention(dim=512, num_heads=2)

        #
        self.AttentionModule = AttentionModule(in_channels=512)

        # 定义多个MLP模块，每个MLP处理不同层的特征
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim * 3, dim // decay, 1, bias=False),
                nn.BatchNorm2d(dim // decay),
                nn.ReLU(),
                nn.Conv2d(dim // decay, dim // decay, 3, 1, padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(dim // decay, dim // decay, 3, 1, padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(dim // decay, dim, 3, 1, padding=1, bias=False)
            ) for dim in in_dim
        ])

        # 增强的分类器头部
        self.conv1 = nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(1024)
        self.relu = nn.ReLU(inplace=True)
        self.res_block = ResidualBlock(1024, 512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_class)

    def forward(self, fx, fy):
        c2x, c3x, c4x, c5x = fx

        c5x, c5y = self.structure_enhance(fx, fy)

        # 下采样到目标大小 [16, 384, 14, 14]
        c2f = self.down_c2(c2x)
        c3f = self.down_c3(c3x)
        c4f = self.down_c4(c4x)
        # print(f'Initial shapes - c5x: {c5x.shape}, c4x: {c4f.shape}, c3x: {c3f.shape}, c2x: {c2f.shape}')

        # # 自注意力机制
        # attn_output = self.self_attention(c4f, c3f, c2f)
        # # 自注意力消融实验
        # # attn_output = self.self_attention(c2f, c3f, c4f)
        # # 将注意力特征与增强后的特征进行融合
        # fused_feature = attn_output + c5x

        # 通道注意力+空间注意力
        fused_feature = c2f + c3f + c4f + c5x
        fused_feature = self.AttentionModule(fused_feature)

        # 增强分类器头部的前向传播
        c5x = self.conv1(fused_feature)
        c5x = self.bn1(c5x)
        c5x = self.relu(c5x)
        c5z = self.res_block(c5x)
        c5x = self.avgpool(c5z)
        c5y = torch.flatten(c5x, 1)
        pred_x = self.fc(c5y)

        return c5y

class ClassifierNIA(nn.Module):
    """增强的分类器模块"""
    def __init__(self, in_dim=[64, 128, 256, 512], num_class=6, decay=4):
        super().__init__()
        c2_channel, c3_channel, c4_channel, c5_channel = in_dim

        self.structure_enhance = FeatureInjector(dim1=c5_channel)

        self.down_c2 = nn.Sequential(
            nn.Conv2d(c2_channel, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.down_c3 = nn.Sequential(
            nn.Conv2d(c3_channel, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.down_c4 = nn.Sequential(
            nn.Conv2d(c4_channel, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # 自注意力机制
        self.self_attention = SelfAttention(dim=512, num_heads=2)

        #
        self.AttentionModule = AttentionModule(in_channels=512)

        # 定义多个MLP模块，每个MLP处理不同层的特征
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim * 3, dim // decay, 1, bias=False),
                nn.BatchNorm2d(dim // decay),
                nn.ReLU(),
                nn.Conv2d(dim // decay, dim // decay, 3, 1, padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(dim // decay, dim // decay, 3, 1, padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(dim // decay, dim, 3, 1, padding=1, bias=False)
            ) for dim in in_dim
        ])

        # 增强的分类器头部
        self.conv1 = nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(1024)
        self.relu = nn.ReLU(inplace=True)
        self.res_block = ResidualBlock(1024, 512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_class)

    def forward(self, fx, fy):
        c2x, c3x, c4x = fx
        c2y, c3y, c4y = fy

        # c5x, c5y = self.structure_enhance(fx, fy)

        # 下采样到目标大小 [16, 384, 14, 14]
        c2f = self.down_c2(c2x)
        c3f = self.down_c3(c3x)
        c4f = self.down_c4(c4x)
        # print(f'Initial shapes - c5x: {c5x.shape}, c4x: {c4f.shape}, c3x: {c3f.shape}, c2x: {c2f.shape}')

        # 通道注意力+空间注意力
        fused_feature = c2f + c3f + c4f
        fused_feature = self.AttentionModule(fused_feature)

        # 增强分类器头部的前向传播
        c5x = self.conv1(fused_feature)
        c5x = self.bn1(c5x)
        c5x = self.relu(c5x)
        c5x = self.res_block(c5x)
        c5x = self.avgpool(c5x)
        c5x = torch.flatten(c5x, 1)
        pred_x = self.fc(c5x)

        return pred_x

class ClassifierNHFE(nn.Module):
    """增强的分类器模块"""
    def __init__(self, in_dim=[64, 128, 256, 512], num_class=6, decay=4):
        super().__init__()
        c2_channel, c3_channel, c4_channel, c5_channel = in_dim

        self.structure_enhance = FeatureInjector(dim1=c5_channel)

        self.down_c2 = nn.Sequential(
            nn.Conv2d(c2_channel, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.down_c3 = nn.Sequential(
            nn.Conv2d(c3_channel, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.down_c4 = nn.Sequential(
            nn.Conv2d(c4_channel, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # 自注意力机制
        self.self_attention = SelfAttention(dim=512, num_heads=2)

        #
        self.AttentionModule = AttentionModule(in_channels=512)

        # 定义多个MLP模块，每个MLP处理不同层的特征
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim * 3, dim // decay, 1, bias=False),
                nn.BatchNorm2d(dim // decay),
                nn.ReLU(),
                nn.Conv2d(dim // decay, dim // decay, 3, 1, padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(dim // decay, dim // decay, 3, 1, padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(dim // decay, dim, 3, 1, padding=1, bias=False)
            ) for dim in in_dim
        ])

        # 增强的分类器头部
        self.conv1 = nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(1024)
        self.relu = nn.ReLU(inplace=True)
        self.res_block = ResidualBlock(1024, 512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_class)

    def forward(self, fx, fy):
        c5x = fx

        # c5x, c5y = self.structure_enhance(fx, fy)

        # 下采样到目标大小 [16, 384, 14, 14]
        # c2f = self.down_c2(c2x)
        # c3f = self.down_c3(c3x)
        # c4f = self.down_c4(c4x)
        # print(f'Initial shapes - c5x: {c5x.shape}, c4x: {c4f.shape}, c3x: {c3f.shape}, c2x: {c2f.shape}')

        # 通道注意力+空间注意力
        fused_feature = c5x
        fused_feature = self.AttentionModule(fused_feature)

        # 增强分类器头部的前向传播
        c5x = self.conv1(fused_feature)
        c5x = self.bn1(c5x)
        c5x = self.relu(c5x)
        c5x = self.res_block(c5x)
        c5x = self.avgpool(c5x)
        c5x = torch.flatten(c5x, 1)
        pred_x = self.fc(c5x)

        return pred_x

class ClassifierNCFFA(nn.Module):
    """增强的分类器模块"""
    def __init__(self, in_dim=[64, 128, 256, 512], num_class=6, decay=4):
        super().__init__()
        c2_channel, c3_channel, c4_channel, c5_channel = in_dim

        self.structure_enhance = FeatureInjector(dim1=c5_channel)

        self.down_c2 = nn.Sequential(
            nn.Conv2d(c2_channel, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.down_c3 = nn.Sequential(
            nn.Conv2d(c3_channel, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.down_c4 = nn.Sequential(
            nn.Conv2d(c4_channel, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # 自注意力机制
        self.self_attention = SelfAttention(dim=512, num_heads=2)

        #
        self.AttentionModule = AttentionModule(in_channels=512)

        # 定义多个MLP模块，每个MLP处理不同层的特征
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim * 3, dim // decay, 1, bias=False),
                nn.BatchNorm2d(dim // decay),
                nn.ReLU(),
                nn.Conv2d(dim // decay, dim // decay, 3, 1, padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(dim // decay, dim // decay, 3, 1, padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(dim // decay, dim, 3, 1, padding=1, bias=False)
            ) for dim in in_dim
        ])

        # 增强的分类器头部
        self.conv1 = nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(1024)
        self.relu = nn.ReLU(inplace=True)
        self.res_block = ResidualBlock(1024, 512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_class)

    def forward(self, fx, fy):
        c2x, c3x, c4x, c5x = fx

        # c5x, c5y = self.structure_enhance(fx, fy)

        # 下采样到目标大小 [16, 384, 14, 14]
        c2f = self.down_c2(c2x)
        c3f = self.down_c3(c3x)
        c4f = self.down_c4(c4x)
        # print(f'Initial shapes - c5x: {c5x.shape}, c4x: {c4f.shape}, c3x: {c3f.shape}, c2x: {c2f.shape}')

        # # 自注意力机制
        # attn_output = self.self_attention(c4f, c3f, c2f)
        # # 自注意力消融实验
        # # attn_output = self.self_attention(c2f, c3f, c4f)
        # # 将注意力特征与增强后的特征进行融合
        # fused_feature = attn_output + c5x

        # 通道注意力+空间注意力
        fused_feature = c2f + c3f + c4f + c5x
        fused_feature = self.AttentionModule(fused_feature)

        # 增强分类器头部的前向传播
        c5x = self.conv1(fused_feature)
        c5x = self.bn1(c5x)
        c5x = self.relu(c5x)
        c5x = self.res_block(c5x)
        c5x = self.avgpool(c5x)
        c5x = torch.flatten(c5x, 1)
        pred_x = self.fc(c5x)

        return pred_x

class ClassifierNA(nn.Module):
    """增强的分类器模块"""
    def __init__(self, in_dim=[64, 128, 256, 512], num_class=6, decay=4):
        super().__init__()
        c2_channel, c3_channel, c4_channel, c5_channel = in_dim

        self.structure_enhance = FeatureInjector(dim1=c5_channel)

        self.down_c2 = nn.Sequential(
            nn.Conv2d(c2_channel, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.down_c3 = nn.Sequential(
            nn.Conv2d(c3_channel, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.down_c4 = nn.Sequential(
            nn.Conv2d(c4_channel, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # 自注意力机制
        self.self_attention = SelfAttention(dim=512, num_heads=2)

        #
        self.AttentionModule = AttentionModule(in_channels=512)

        # 定义多个MLP模块，每个MLP处理不同层的特征
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim * 3, dim // decay, 1, bias=False),
                nn.BatchNorm2d(dim // decay),
                nn.ReLU(),
                nn.Conv2d(dim // decay, dim // decay, 3, 1, padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(dim // decay, dim // decay, 3, 1, padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(dim // decay, dim, 3, 1, padding=1, bias=False)
            ) for dim in in_dim
        ])

        # 增强的分类器头部
        self.conv1 = nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(1024)
        self.relu = nn.ReLU(inplace=True)
        self.res_block = ResidualBlock(1024, 512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_class)

    def forward(self, fx, fy):
        c2x, c3x, c4x, c5x = fx

        c5x, c5y = self.structure_enhance(fx, fy)

        # 下采样到目标大小 [16, 384, 14, 14]
        c2f = self.down_c2(c2x)
        c3f = self.down_c3(c3x)
        c4f = self.down_c4(c4x)
        # print(f'Initial shapes - c5x: {c5x.shape}, c4x: {c4f.shape}, c3x: {c3f.shape}, c2x: {c2f.shape}')

        # # 自注意力机制
        # attn_output = self.self_attention(c4f, c3f, c2f)
        # # 自注意力消融实验
        # # attn_output = self.self_attention(c2f, c3f, c4f)
        # # 将注意力特征与增强后的特征进行融合
        # fused_feature = attn_output + c5x

        # 通道注意力+空间注意力
        fused_feature = c2f + c3f + c4f + c5x
        # fused_feature = self.AttentionModule(fused_feature)

        # 增强分类器头部的前向传播
        c5x = self.conv1(fused_feature)
        c5x = self.bn1(c5x)
        c5x = self.relu(c5x)
        c5x = self.res_block(c5x)
        c5x = self.avgpool(c5x)
        c5x = torch.flatten(c5x, 1)
        pred_x = self.fc(c5x)

        return pred_x

class Classifier1(nn.Module):
    """增强的分类器模块"""
    def __init__(self, in_dim=[64, 128, 256, 512], num_class=6, decay=4):
        super().__init__()
        c2_channel, c3_channel, c4_channel, c5_channel = in_dim

        self.structure_enhance = FeatureInjector(dim1=c5_channel)

        # 替换下采样操作为上采样操作，从低分辨率逐步上采样到高分辨率
        self.upc5 = nn.Sequential(
            nn.Conv2d(c5_channel, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        self.upc4 = nn.Sequential(
            nn.Conv2d(512 + c4_channel, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        self.upc3 = nn.Sequential(
            nn.Conv2d(256 + c3_channel, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        self.upc2 = nn.Sequential(
            nn.Conv2d(128 + c2_channel, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 自注意力机制
        self.self_attention = SelfAttention(dim=64, num_heads=2)

        # 通道注意力模块
        self.AttentionModule = AttentionModule(in_channels=c5_channel)  # 最终特征通道数应与最细粒度特征匹配

        # 增强的分类器头部
        self.conv1 = nn.Conv2d(192, 1024, kernel_size=3, padding=1, bias=False)  # 调整输入通道数为 192
        self.bn1 = nn.BatchNorm2d(1024)
        self.relu = nn.ReLU(inplace=True)
        self.res_block = ResidualBlock(1024, 512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_class)

    def forward(self, fx, fy):
        c2x, c3x, c4x, c5x = fx

        c5x, c5y = self.structure_enhance(fx, fy)

        c5x = self.AttentionModule(c5x)

        # 上采样并与浅层特征进行concat
        c5_up = self.upc5(c5x)
        c4f = torch.cat([c5_up, c4x], dim=1)  # concat along channel dimension
        c4_up = self.upc4(c4f)
        c3f = torch.cat([c4_up, c3x], dim=1)
        c3_up = self.upc3(c3f)
        c2f = torch.cat([c3_up, c2x], dim=1)

        # # 上采样后的特征经过自注意力和分类器头部
        # c2f = self.upc2(c2f)  # Final feature refinement
        # c2f = self.self_attention(c2f)  # Optional: use attention

        # 增强分类器头部的前向传播
        c2f = self.conv1(c2f)
        c2f = self.bn1(c2f)
        c2f = self.relu(c2f)
        c2f = self.res_block(c2f)
        c2f = self.avgpool(c2f)
        c2f = torch.flatten(c2f, 1)
        pred_x = self.fc(c2f)

        return pred_x



class FeatureVisionModel(nn.Module):
    """增强的分类器模块"""
    def __init__(self, in_dim=[64, 128, 256, 512], num_class=6, decay=4):
        super().__init__()
        c2_channel, c3_channel, c4_channel, c5_channel = in_dim

        self.structure_enhance = FeatureInjector(dim1=c5_channel)

        self.down_c2 = nn.Sequential(
            nn.Conv2d(c2_channel, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.down_c3 = nn.Sequential(
            nn.Conv2d(c3_channel, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.down_c4 = nn.Sequential(
            nn.Conv2d(c4_channel, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # 自注意力机制
        self.self_attention = SelfAttention(dim=512, num_heads=2)

        #
        self.AttentionModule = AttentionModule(in_channels=512)

        # 定义多个MLP模块，每个MLP处理不同层的特征
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim * 3, dim // decay, 1, bias=False),
                nn.BatchNorm2d(dim // decay),
                nn.ReLU(),
                nn.Conv2d(dim // decay, dim // decay, 3, 1, padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(dim // decay, dim // decay, 3, 1, padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(dim // decay, dim, 3, 1, padding=1, bias=False)
            ) for dim in in_dim
        ])

        # 增强的分类器头部
        self.conv1 = nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(1024)
        self.relu = nn.ReLU(inplace=True)
        self.res_block = ResidualBlock(1024, 512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_class)

    def forward(self, fx, fy):
        c2x, c3x, c4x, c5x = fx

        c5x, c5y = self.structure_enhance(fx, fy)

        # 下采样到目标大小 [16, 384, 14, 14]
        c2f = self.down_c2(c2x)
        c3f = self.down_c3(c3x)
        c4f = self.down_c4(c4x)
        # print(f'Initial shapes - c5x: {c5x.shape}, c4x: {c4f.shape}, c3x: {c3f.shape}, c2x: {c2f.shape}')

        # # 自注意力机制
        # attn_output = self.self_attention(c4f, c3f, c2f)
        # # 自注意力消融实验
        # # attn_output = self.self_attention(c2f, c3f, c4f)
        # # 将注意力特征与增强后的特征进行融合
        # fused_feature = attn_output + c5x

        # 通道注意力+空间注意力
        fused_feature = c2f + c3f + c4f + c5x
        fused_feature = self.AttentionModule(fused_feature)

        # 增强分类器头部的前向传播
        # c5x = self.conv1(fused_feature)
        # c5x = self.bn1(c5x)
        # c5x = self.relu(c5x)
        # c5x = self.res_block(c5x)
        # c5x = self.avgpool(c5x)
        # c5x = torch.flatten(c5x, 1)
        # pred_x = self.fc(c5x)

        return c3x
