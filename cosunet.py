import torch.nn as nn
import torch
import torch.nn.functional as F

class UNetConvBlock(nn.Module):

    def __init__(self, in_size, out_size):
        super(UNetConvBlock, self).__init__()
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

class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()

        # select the upsampling mode
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),)

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y: (diff_y + target_size[0]), diff_x: (diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out


class Unet(nn.Module):

    def __init__(self, in_channels=1, depth=5, wf=6, up_mode='upconv'):
        super(Unet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.depth = depth
        prev_channels = in_channels

        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels, 2 ** (wf + i)))
            prev_channels = 2 ** (wf + i)

#        self.up_path = nn.ModuleList()
#        for i in reversed(range(depth - 1)):
#            self.up_path.append(UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm))
#            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, prev_channels, kernel_size=1)

    def forward(self, x):
        #  blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
          #  if i != len(self.down_path) - 1:
          #      blocks.append(x)
          #      x_temp = F.max_pool2d(x, 2)

        #for i, up in enumerate(self.up_path):
           # x = up(x, blocks[-i - 1])
        return self.last(x)


class CoattentionModel(nn.Module):
    def __init__(self, initial_channel, num_classes, all_channel=256):
        super(CoattentionModel, self).__init__()
        self.encoder = Unet(initial_channel, 4, 5)
        self.linear_e = nn.Linear(all_channel, all_channel, bias=False)
        self.channel = all_channel
        self.gate = nn.Conv2d(all_channel, 1, kernel_size=1, bias=False)
        self.gate_s = nn.Sigmoid()
        self.conv1 = nn.Conv2d(all_channel*2, all_channel, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(all_channel*2, all_channel, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(all_channel)
        self.bn2 = nn.BatchNorm2d(all_channel)
        self.prelu = nn.ReLU(inplace=True)
        self.main_classifier1 = nn.Conv2d(all_channel, num_classes, kernel_size=1, bias=True)
        self.main_classifier2 = nn.Conv2d(all_channel, num_classes, kernel_size=1, bias=True)
        self.softmax = nn.Sigmoid()

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

        x1 = self.main_classifier1(input1_att)
        x2 = self.main_classifier2(input2_att)

        x1 = F.upsample(x1, input_size, mode='bilinear')  # upsample to the size of input image, scale=8
        x2 = F.upsample(x2, input_size, mode='bilinear')  # upsample to the size of input image, scale=8

        x1 = self.softmax(x1)
        x2 = self.softmax(x2)

        return x1, x2


class get_module(nn.Module):

    def __init__(self, initial_channel, num_classes):
        super(get_module, self).__init__()
        self.model = CoattentionModel(initial_channel, num_classes)

    def forward(self, x1, x2):
        return self.model(x1, x2)
