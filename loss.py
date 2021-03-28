import torch.nn as nn
import torch
import numpy as np

class Content_Loss(nn.Module):
    def __init__(self, target, weight):
        super(Content_Loss, self).__init__()
        self.weight = weight
        self.target = target.detach() * self.weight
        # 必须要用detach来分离出target，这时候target不再是一个Variable，这是为了动态计算梯度，否则forward会出错，不能向前传播
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.loss = self.criterion(input * self.weight, self.target)
        out = input.clone()
        return out

    def backward(self, retain_variabels=True):
        self.loss.backward(retain_graph=retain_variabels)
        return self.loss

class Gram(nn.Module):
    def __init__(self):
        super(Gram, self).__init__()

    def forward(self, input):
        a, b, c, d = input.size()
        feature = input.view(a * b, c * d)
        gram = torch.mm(feature, feature.t())
        gram /= (a * b * c * d)

        return gram


class Style_Loss(nn.Module):
    def __init__(self, target, weight, content_masks, style_masks, loss_list):
        super(Style_Loss, self).__init__()
        self.weight = weight
        #self.target = target.detach() * self.weight
        self.target = target.detach()
        self.gram = Gram()
        self.criterion = nn.MSELoss()
        self.content_masks = content_masks
        self.style_masks = style_masks
        self.num_channels = self.target.shape[1]
        self.loss_list = loss_list

    def forward(self, input):
        # new_input = input.clone()
        # new_target = self.target.clone()

        # for index in range(input.shape[1]):
        #     new_input[0][index] = torch.mul(input[0][index], self.content_masks[3])
        self.loss = 0
        for seg_index in self.loss_list:
            new_input = input.mul(self.content_masks[seg_index]).clone()
            tmpG = self.gram(new_input) * self.weight
            style_mask_mean = torch.mean(self.style_masks[seg_index])
            tmpG = tmpG / style_mask_mean

            new_target = self.target.mul(self.style_masks[seg_index]).clone()
            tmpTar = self.gram(new_target) * self.weight
            content_mask_mean = torch.mean(self.content_masks[seg_index])
            tmpTar = tmpTar / content_mask_mean

            diff_style_sum = self.criterion(tmpG, tmpTar) * content_mask_mean
            self.loss = self.loss + diff_style_sum


        #self.loss = self.criterion(G, self.target)
        out = input.clone()
        return out

    def backward(self, retain_variabels=True):
        self.loss.backward(retain_graph=retain_variabels)
        return self.loss
