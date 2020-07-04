import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 输入D*D的图像，输出D*D的图像
# class BlockNet(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, out_activation=nn.ReLU):
#         super(BlockNet, self).__init__()
#         self.res1 = ResUnit(in_channels, in_channels, kernel_size=kernel_size)
#         self.res2 = ResUnit(in_channels, out_channels, kernel_size=kernel_size, 
#             padding=padding, stride=stride, out_activation=out_activation)

#     def forward(self, x):
#         out = self.res1(x)
#         out = self.res2(out)
#         return out

class ResUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                stride=1, padding=1, out_activation=nn.ReLU):
        super(ResUnit, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding)
        # if (in_channels != out_channels):
        #     self.projection = nn.Conv2d(
        #         in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        if out_activation:
            self.out_activation = out_activation()

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        # if (x.shape[1] != out.shape[1]):
        #     x = self.projection(x)
        # out = out + x
        if hasattr(self, 'out_activation'):
            out = self.out_activation(out)
        return out