import torch
from torch import nn
from einops import rearrange, repeat

# 定义一个预归一化模块
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)  # 使用 LayerNorm 对输入进行归一化
        self.fn = fn  # 传入的函数（如注意力模块或前馈网络）

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)  # 先对输入进行归一化，然后应用传入的函数

# 定义一个前馈网络模块
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),  # 第一层线性变换
            nn.GELU(),  # 使用 GELU 激活函数
            nn.Dropout(dropout),  # Dropout 层
            nn.Linear(hidden_dim, dim),  # 第二层线性变换
            nn.Dropout(dropout)  # Dropout 层
        )

    def forward(self, x):
        return self.net(x)  # 前向传播，返回处理后的数据

# 定义一个多头自注意力模块
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads  # 计算内部维度
        project_out = not (heads == 1 and dim_head == dim)  # 判断是否需要输出投影

        self.heads = heads  # 头的数量
        self.scale = dim_head ** -0.5  # 缩放因子

        self.attend = nn.Softmax(dim=-1)  # 使用 Softmax 计算注意力权重
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)  # 将输入投影到 Q、K、V

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),  # 输出投影
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()  # 如果不需要输出投影，则使用恒等映射

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 将 Q、K、V 分离
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)  # 重新排列 Q、K、V 的形状
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # 计算点积并应用缩放因子
        attn = self.attend(dots)  # 计算注意力权重
        out = torch.matmul(attn, v)  # 计算加权和
        out = rearrange(out, 'b h n d -> b n (h d)')  # 重新排列输出的形状
        return self.to_out(out), attn  # 返回输出和注意力权重

# 定义一个 Transformer 模块
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])  # 初始化一个模块列表
        for _ in range(depth):  # 根据深度参数构建多层 Transformer
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),  # 注意力模块
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))  # 前馈网络模块
            ]))

    def forward(self, x):
        att = None  # 初始化注意力权重
        for attn, ff in self.layers:  # 遍历每一层
            att_x, att = attn(x)  # 应用注意力模块
            x = att_x + x  # 残差连接
            x = ff(x) + x  # 应用前馈网络模块并添加残差连接
        return x, att.mean(dim=1)  # 返回输出和平均注意力权重

# class WordEmbTransformers(nn.Module):
#     def __init__(self, feature_dim, dropout):
#         super(WordEmbTransformers, self).__init__()
#         self.feature_dim = feature_dim
#         self.dropout = dropout
#         self.fc = nn.Sequential(nn.Linear(in_features=768,
#                                           out_features=128,
#                                           bias=True),
#                                 nn.ReLU(),
#                                 nn.Dropout(p=self.dropout),
#                                 nn.Linear(in_features=128,
#                                           out_features=self.feature_dim,
#                                           bias=True)
#                                 )
#
#     def forward(self, x):
#         # 0-1
#         x = self.fc(x)
#         return x


class WordEmbTransformers(nn.Module):
    def __init__(self, feature_dim, dropout):
        super(WordEmbTransformers, self).__init__()
        self.feature_dim = feature_dim
        self.dropout = dropout

        self.fc = nn.Sequential(
            nn.Linear(in_features=768, out_features=256, bias=True),
            nn.BatchNorm1d(256),  # 添加 Batch Normalization
            nn.GELU(),  # 使用 GELU 激活函数
            nn.Dropout(p=self.dropout),

            nn.Linear(in_features=256, out_features=128, bias=True),
            nn.BatchNorm1d(128),  # 添加 Batch Normalization
            nn.GELU(),  # 使用 GELU 激活函数
            nn.Dropout(p=self.dropout),

            nn.Linear(in_features=128, out_features=self.feature_dim, bias=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x