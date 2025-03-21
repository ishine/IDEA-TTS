import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import models.commons as commons


class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            # nn.BatchNorm1d(bottleneck), # I remove this layer
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
            )

    def forward(self, input):
        x = self.se(input)
        return input * x


class Bottle2neck(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale = 8):
        super(Bottle2neck, self).__init__()
        width       = int(math.floor(planes / scale))
        self.conv1  = nn.Conv1d(inplanes, width*scale, kernel_size=1)
        self.bn1    = nn.BatchNorm1d(width*scale)
        self.nums   = scale -1
        convs       = []
        bns         = []
        num_pad = math.floor(kernel_size/2)*dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs  = nn.ModuleList(convs)
        self.bns    = nn.ModuleList(bns)
        self.conv3  = nn.Conv1d(width*scale, planes, kernel_size=1)
        self.bn3    = nn.BatchNorm1d(planes)
        self.relu   = nn.ReLU()
        self.width  = width
        self.se     = SEModule(planes)

    def forward(self, x, x_mask):
        residual = x * x_mask
        out = self.conv1(x * x_mask)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i==0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp * x_mask)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            if i==0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]),1)

        out = self.conv3(out * x_mask)
        out = self.relu(out)
        out = self.bn3(out)
        
        out = self.se(out * x_mask)
        out += residual
        return out * x_mask


class Attention(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck=128):
        super(Attention, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, bottleneck, kernel_size=1)
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm1d(bottleneck)
        self.tanh = nn.Tanh()
        self.conv2 = nn.Conv1d(bottleneck, out_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x, x_mask):
        x = self.conv1(x * x_mask)
        x = self.relu(x)
        x = self.norm(x)
        x = self.tanh(x)
        x = self.conv2(x * x_mask)
        x = self.softmax(x)

        return x * x_mask


class ECAPA_TDNN(nn.Module):
    def __init__(self, input_channel, C, output_channel):
        super(ECAPA_TDNN, self).__init__()
        self.conv1  = nn.Conv1d(input_channel, C, kernel_size=5, stride=1, padding=2)
        self.relu   = nn.ReLU()
        self.bn1    = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(3*C, 3*C, kernel_size=1)
        self.attention = Attention(9*C, 3*C, bottleneck=128)
        self.bn5 = nn.BatchNorm1d(6*C)
        self.fc6 = nn.Linear(6*C, output_channel)
        self.bn6 = nn.BatchNorm1d(output_channel)

    def forward(self, x, x_lengths):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.conv1(x * x_mask)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x * x_mask, x_mask)
        x2 = self.layer2((x+x1) * x_mask, x_mask)
        x3 = self.layer3((x+x1+x2) * x_mask, x_mask)

        x = self.layer4(torch.cat((x1, x2, x3),dim=1))
        x = self.relu(x)
        x = x * x_mask
        t = x.size()[-1]

        global_x = torch.cat((x, torch.mean(x, dim=2, keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        w = self.attention(global_x * x_mask, x_mask)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-4))
        x = torch.cat((mu, sg), 1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)
        return x / torch.norm(x, dim=1, keepdim=True)


def main():
    x = torch.randn(32, 513, 101)
    x_lengths = torch.randn(32)
    spk_enc = ECAPA_TDNN(513, 512, 256)

    print(spk_enc(x, x_lengths).size())


if __name__ == '__main__':
    main()