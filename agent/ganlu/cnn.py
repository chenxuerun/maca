import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 输入D*D的图像，输出D*D的图像
# 保证不改变feature_map

class BlockNet(nn.Module):
    def __init__(self, in_channel, out_channels, kernel_sizes, out_activation=nn.ReLU):
        super(BlockNet, self).__init__()
        units = []
        last_channel = in_channel
        for i, (out_channel, kernel_size) in enumerate(zip(out_channels[0: -1], kernel_sizes[0: -1])):
            units.append(ResUnit(in_channel=last_channel, out_channel=out_channel, kernel_size=kernel_size))
            last_channel = out_channel
        units.append(ResUnit(in_channel=last_channel, out_channel=out_channels[-1], 
            kernel_size=kernel_sizes[-1], out_activation=out_activation))
        self.net = nn.Sequential(*units)

    def forward(self, input):
        return self.net(input)

class ResUnit(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, out_activation=nn.ReLU):
        super(ResUnit, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, (in_channel + out_channel) // 2, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.conv2 = nn.Conv2d((in_channel + out_channel) // 2, out_channel, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        if (in_channel != out_channel):
            self.projection = nn.Conv2d(
                in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False)
        if out_activation:
            self.out_activation = out_activation()

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        if (x.shape[1] != out.shape[1]):
            x = self.projection(x)
        out = out + x
        if hasattr(self, 'out_activation'):
            out = self.out_activation(out)
        return out