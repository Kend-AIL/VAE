import torch
import torch.nn as nn


def nonlinearity(x):
    # Swish
    return x*torch.sigmoid(x)
class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        self.upsample=torch.nn.Upsample(scale_factor=2.0, mode="nearest")
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = self.upsample(x)
        if self.with_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        return x



class SimpleEncoder(nn.Module):
    def __init__(self, in_channels, resolution, ch, ch_mult):
        super(SimpleEncoder, self).__init__()
        self.resolution = resolution
        self.ch = ch
        self.down = nn.ModuleList()
        self.num_resolutions = len(ch_mult)
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)
        self.activate=nonlinearity
        self.conv=nn.ModuleList()
        self.bn=nn.ModuleList()
        self.res=nn.ModuleList()
        self.attn=nn.ModuleList()
        in_ch_mult = (1,) + tuple(ch_mult)
        for i in range(self.num_resolutions):
            block_in = ch * in_ch_mult[i]
            block_out = ch * ch_mult[i]
            self.bn.append(nn.BatchNorm2d(block_in))
            self.res.append(ResidualBlock(block_in))
            conv=nn.Conv2d(block_in, block_out, kernel_size=3, stride=1, padding=1)
            down = Downsample(block_out, with_conv=True)
            self.down.append(down)
            self.conv.append(conv)
            self.attn.append(AttnBlock(block_out))
            

    def forward(self, x):
        h = self.conv_in(x)
        for i, (down, conv) in enumerate(zip(self.down, self.conv)):
            h = self.bn[i](h)
            h = self.res[i](h)
            h=self.activate(h)
            h = self.activate(conv(h))
            h=self.attn[i](h)
            h=self.activate(h)
            h = self.activate(down(h))  
        return h

class SimpleDecoder(nn.Module):
    
    def __init__(self, in_channels, resolution, ch, ch_mult):
        super(SimpleDecoder, self).__init__()
        self.resolution = resolution
        self.ch = ch
        self.up = nn.ModuleList()
        self.conv = nn.ModuleList()  # Additional conv layers
        self.num_resolutions = len(ch_mult)
        self.activate = nonlinearity
        self.bn=nn.ModuleList()
        self.res=nn.ModuleList()
        self.attn=nn.ModuleList()
        in_ch_mult = ch_mult[::-1]
        reverse_ch=tuple(in_ch_mult)+(1,)
        out_ch_mult=reverse_ch[1:]
        for i in range(self.num_resolutions):
            block_in = ch * in_ch_mult[i]
            block_out = ch * out_ch_mult[i]
            conv_layer = nn.Conv2d(block_in, block_out, kernel_size=3, stride=1, padding=1)
            self.conv.append(conv_layer)
            up = Upsample(block_out, with_conv=True)
            self.up.append(up)
            self.bn.append(nn.BatchNorm2d(block_in))
            self.res.append(ResidualBlock(block_in))
            self.attn.append(AttnBlock(block_out))
        self.conv_out = nn.Conv2d(ch, in_channels, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        h = x
        for i, (up, conv) in enumerate(zip(self.up, self.conv)):
            h = self.bn[i](h)
            h = self.res[i](h)
            h=self.activate(h)
            h = self.activate(conv(h))  
            h= self.attn[i](h)
            h=self.activate(h)
            h = self.activate(up(h))    # 合并激活函数和上采样层

        h = self.conv_out(h)
        h = torch.tanh(h)
        return h

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # skip connection
        out = self.relu(out)
        return out
    

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        # 定义查询（query），键（key），值（value）以及输出投影的卷积层
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        # 计算查询（query），键（key），值（value）张量
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # 为注意力计算重塑和置换查询（query），键（key），值（value）
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)  # b, hw, c
        k = k.reshape(b, c, h*w)  # b, c, hw
        w_ = torch.bmm(q, k)  # b, hw, hw; w[b, i, j] = sum_c q[b, i, c] * k[b, c, j]
        w_ = w_ * (int(c)**(-0.5))  # 缩放注意力分数
        w_ = torch.nn.functional.softmax(w_, dim=2)  # 对注意力分数应用softmax

        # 计算值（value）的加权和
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)  # b, hw, hw
        h_ = torch.bmm(v, w_)  # b, c, hw; h_[b, c, j] = sum_i
        h_ = h_.reshape(b, c, h, w)

        # 应用输出投影并加上残差连接
        h_ = self.proj_out(h_)

        return x + h_  # 将输入和输出相加以形成残差连接
