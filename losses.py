import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLossMulticlass_CW(nn.Module):
    def __init__(self):
        super(DiceLossMulticlass_CW, self).__init__()
        self.smooth = 1e-5

    def forward(self, inputs, targets, n_classes, weights=None, validation=False):

        if weights is not None:
            weights = weights / weights.sum()

        inputs = inputs.permute(0,2,1).contiguous().view(-1, inputs.size(1))
        targets = targets.permute(0,2,1).contiguous().view(-1, targets.size(1)).long()

        N, C = inputs.size()
        prob = torch.softmax(inputs, dim=1)
        t_one_hot = inputs.new_zeros(inputs.size())
        t_one_hot.scatter_(1, targets, 1.)

        if weights is None:
            iflat = prob.contiguous().view(-1)
            tflat = t_one_hot.contiguous().view(-1)
            intersection = (iflat * tflat).sum()
            return 1 - ((2. * intersection) / (iflat.sum() + tflat.sum() + self.smooth))
        else:
            intersection = (prob * t_one_hot).sum(dim=0)
            summ = prob.sum(dim=0) + t_one_hot.sum(dim=0)
            loss = 1 - ((2. * intersection) / (summ + self.smooth))
            weight = weights.type_as(prob)
            loss *= weight
            return loss.mean()


class CrossEntropy(nn.Module):
    def __init__(self, topk_rate=1.0):
        super(CrossEntropy, self).__init__()
        self.topk_rate = topk_rate

    def forward(self, output, target, n_classes, weights, softmaxed=False):
        target = torch.squeeze(target, 1)
        wce = F.cross_entropy(output, target.long(), weight=weights)
        return wce

class FocalLoss(nn.Module):

    def __init__(self, ignore_index,focusing_param=2):
        super(FocalLoss, self).__init__()

        self.focusing_param = focusing_param
        self.ignore_index = ignore_index

    def forward(self, output, target, n_classes, weights=None):

        target = torch.squeeze(target, 1)
        output = F.log_softmax(output, 1)
        logpt = -F.nll_loss(output, target.long(), ignore_index=self.ignore_index)
        pt = torch.exp(logpt)

        if weights is not None:
            weighted_logpt = -F.nll_loss(output, target.long(), weights, ignore_index=self.ignore_index)
            focal_loss = -((1 - pt) ** self.focusing_param) * weighted_logpt
        else:
            focal_loss = -((1 - pt) ** self.focusing_param) * logpt

        return focal_loss


class FocalLoss_Pixel(nn.Module):

    def __init__(self, focusing_param=2, balanced_param=1):
        super(FocalLoss_Pixel, self).__init__()

        self.focusing_param = focusing_param
        self.balanced_param = balanced_param

    def forward(self, output, target, n_classes, weights, softmaxed=False):

        if not softmaxed:
            output = F.log_softmax(output, 1)

        target = torch.squeeze(target, 1).long()

        logpt = -F.nll_loss(output, target, weights, reduction='none').view(-1)
        pt = torch.exp(logpt)

        focal_loss = -((1 - pt) ** self.focusing_param) * logpt

        w = weights.index_select(0, target.view(-1)).sum()

        wf = focal_loss.sum() / w

        return self.balanced_param * wf

def softmax_mse_loss(input_logits, target_logits):
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    mse_loss = (input_softmax - target_softmax) ** 2
    return mse_loss

def softmax_kl_loss(input_logits, target_logits):
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.kl_divider(input_log_softmax, target_softmax, size_average=False)