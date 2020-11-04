import torch.nn as nn
import torch
import torch.nn.functional as F

class UNetDownBlock(nn.Module):

    def __init__(self, in_size, out_size):
        super(UNetDownBlock, self).__init__()
        block = []
        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=1))
        block.append(nn.ReLU())
        block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=1))
        block.append(nn.ReLU())
        block.append(nn.BatchNorm2d(out_size))

        block.append(nn.MaxPool2d(2))

        self.block = nn.Sequential(*block)  # the elements in the list are seen as independent params due to *

    def forward(self, x):
        out = self.block(x)
        return out


class UnetEncoder(nn.Module):
    def __init__(self, in_channels=1, depth=5, wf=6):
        super(UnetEncoder, self).__init__()
        self.depth = depth
        prev_channels = in_channels

        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(UNetDownBlock(prev_channels, 2 ** (wf + i)))
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, prev_channels, kernel_size=1)

    def forward(self, x):
        for i, down in enumerate(self.down_path):
            x = down(x)
        return self.last(x)


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUpBlock, self).__init__()
        self.up = nn.Sequential(nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2), nn.ReLU(), nn.BatchNorm2d(out_size))
        self.conv_block = nn.Sequential(nn.Conv2d(out_size, out_size, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(out_size))

    def forward(self, x):
        x = self.up(x)
        out = self.conv_block(x)
        return out


class UnetDecoder(nn.Module):
    def __init__(self, in_channels=1, depth=5, wf=6, n_classes=4):
        super(UnetDecoder, self).__init__()
        self.depth = depth
        prev_channels = in_channels

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, 2 ** (wf + i)))
            prev_channels = 2 ** (wf + i)

        self.upsample = nn.Sequential(nn.ConvTranspose2d(prev_channels, prev_channels, kernel_size=2, stride=2), nn.ReLU(), nn.BatchNorm2d(prev_channels))
        self.last = nn.Sequential(nn.Conv2d(prev_channels, n_classes, kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        for i, up in enumerate(self.up_path):
            x = up(x)
        x = self.upsample(x)
        return self.last(x)


class CoattentionModel(nn.Module):
    def __init__(self, initial_channel, num_classes, all_channel=256, depth=5, wf=4):
        super(CoattentionModel, self).__init__()
        self.encoder = UnetEncoder(initial_channel, depth, wf)
        self.linear_e = nn.Linear(all_channel, all_channel, bias=False)
        self.channel = all_channel
        self.gate = nn.Conv2d(all_channel, 1, kernel_size=1, bias=False)
        self.gate_s = nn.Sigmoid()
        self.conv1 = nn.Conv2d(all_channel*2, all_channel, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(all_channel*2, all_channel, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(all_channel)
        self.bn2 = nn.BatchNorm2d(all_channel)
        self.prelu = nn.ReLU(inplace=True)
        self.decoder1 = UnetDecoder(all_channel, depth, wf, num_classes)
        self.decoder2 = UnetDecoder(all_channel, depth, wf, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input1, input2):  # 注意input2 可以是多帧图像
        input_size = input1.size()[2:]

        exemplar = self.encoder(input1)
        query = self.encoder(input2)

        fea_size = query.size()[2:]
        all_dim = fea_size[0]*fea_size[1]

        exemplar_flat = exemplar.view(-1, query.size()[1], all_dim)  # N,C,H*W
        query_flat = query.view(-1, query.size()[1], all_dim)

        exemplar_t = torch.transpose(exemplar_flat,1,2).contiguous()  # batch size x dim x num
        exemplar_corr = self.linear_e(exemplar_t)

        A = torch.bmm(exemplar_corr, query_flat)
        A1 = F.softmax(A.clone(), dim = 1)
        B = F.softmax(torch.transpose(A,1,2),dim=1)

        query_att = torch.bmm(exemplar_flat, A1).contiguous()  # 注意我们这个地方要不要用交互以及Residual的结构
        exemplar_att = torch.bmm(query_flat, B).contiguous()

        input1_att = exemplar_att.view(-1, query.size()[1], fea_size[0], fea_size[1])
        input2_att = query_att.view(-1, query.size()[1], fea_size[0], fea_size[1])

        input1_mask = self.gate(input1_att)
        input2_mask = self.gate(input2_att)

        input1_mask = self.gate_s(input1_mask)
        input2_mask = self.gate_s(input2_mask)

        input1_att = input1_att * input1_mask
        input2_att = input2_att * input2_mask

        input1_att = torch.cat([input1_att, exemplar],1)
        input2_att = torch.cat([input2_att, query],1)

        input1_att  = self.conv1(input1_att)
        input2_att  = self.conv2(input2_att)

        input1_att  = self.bn1(input1_att)
        input2_att  = self.bn2(input2_att)

        input1_att  = self.prelu(input1_att)
        input2_att  = self.prelu(input2_att)

        x1 = self.decoder1(input1_att)
        x2 = self.decoder2(input2_att)

        return x1, x2


class get_module(nn.Module):

    def __init__(self, initial_channel, num_classes):
        super(get_module, self).__init__()
        self.model = CoattentionModel(initial_channel, num_classes)

    def forward(self, x1, x2):
        return self.model(x1, x2)
