import torch.nn as nn
import torch
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, channel_dim, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_per_head = channel_dim // num_heads
        
        self.query_transform = nn.Linear(channel_dim, channel_dim)
        self.key_transform = nn.Linear(channel_dim, channel_dim)
        self.value_transform = nn.Linear(channel_dim, channel_dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, guide, hidden_rep):
        batch_size, channels, height, width = hidden_rep.size()

        # 将通道维度分割为多个头
        def split_heads(x):
            return x.view(batch_size, self.num_heads, self.dim_per_head, height, width).permute(0, 1, 3, 4, 2)

        # 转换 guide 并调整形状以匹配 hidden_rep 的空间维度
        guide = guide.unsqueeze(-1).unsqueeze(-1)  # [b, 128, 1, 1]
        guide = guide.expand(-1, -1, height, width)

        hidden_rep = hidden_rep.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
        guide = guide.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)

        # 变换并分割
        query = split_heads(self.query_transform(guide))
        key = split_heads(self.key_transform(hidden_rep))
        value = split_heads(self.value_transform(hidden_rep))

        # 计算注意力分数
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.dim_per_head ** 0.5)
        weights = self.softmax(scores)

        # 应用注意力权重并组合多头结果
        attended = torch.matmul(weights, value).permute(0, 1, 4, 2, 3).contiguous()
        attended = attended.view(batch_size, channels, height, width)

        return attended