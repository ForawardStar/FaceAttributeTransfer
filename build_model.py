import torch.nn as nn
import torch
import torchvision.models as models
import numpy as np
import loss
import PIL.Image as Image
import math
import cv2
import torch.nn.functional as F

vgg = models.vgg19(pretrained=True).features
vgg = vgg.cuda()

content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def load_seg(content_seg_path,style_seg_path, content_shape, style_shape):
    color_codes = ['BLUE', 'GREEN', 'BLACK', 'WHITE', 'RED', 'YELLOW', 'GREY', 'LIGHT_BLUE', 'PURPLE']

    def _extract_mask(seg, color_str):
        h, w, c = np.shape(seg)
        if color_str == "BLUE":
            mask_r = (seg[:, :, 0] < 0.1).astype(np.uint8)
            mask_g = (seg[:, :, 1] < 0.1).astype(np.uint8)
            mask_b = (seg[:, :, 2] > 0.9).astype(np.uint8)
        elif color_str == "GREEN":
            mask_r = (seg[:, :, 0] < 0.1).astype(np.uint8)
            mask_g = (seg[:, :, 1] > 0.9).astype(np.uint8)
            mask_b = (seg[:, :, 2] < 0.1).astype(np.uint8)
        elif color_str == "BLACK":
            mask_r = (seg[:, :, 0] < 0.1).astype(np.uint8)
            mask_g = (seg[:, :, 1] < 0.1).astype(np.uint8)
            mask_b = (seg[:, :, 2] < 0.1).astype(np.uint8)
        elif color_str == "WHITE":
            mask_r = (seg[:, :, 0] > 0.9).astype(np.uint8)
            mask_g = (seg[:, :, 1] > 0.9).astype(np.uint8)
            mask_b = (seg[:, :, 2] > 0.9).astype(np.uint8)
        elif color_str == "RED":
            mask_r = (seg[:, :, 0] > 0.9).astype(np.uint8)
            mask_g = (seg[:, :, 1] < 0.1).astype(np.uint8)
            mask_b = (seg[:, :, 2] < 0.1).astype(np.uint8)
        elif color_str == "YELLOW":
            mask_r = (seg[:, :, 0] > 0.9).astype(np.uint8)
            mask_g = (seg[:, :, 1] > 0.9).astype(np.uint8)
            mask_b = (seg[:, :, 2] < 0.1).astype(np.uint8)
        elif color_str == "GREY":
            mask_r = np.multiply((seg[:, :, 0] > 0.4).astype(np.uint8),
                                 (seg[:, :, 0] < 0.6).astype(np.uint8))
            mask_g = np.multiply((seg[:, :, 1] > 0.4).astype(np.uint8),
                                 (seg[:, :, 1] < 0.6).astype(np.uint8))
            mask_b = np.multiply((seg[:, :, 2] > 0.4).astype(np.uint8),
                                 (seg[:, :, 2] < 0.6).astype(np.uint8))
        elif color_str == "LIGHT_BLUE":
            mask_r = (seg[:, :, 0] < 0.1).astype(np.uint8)
            mask_g = (seg[:, :, 1] > 0.9).astype(np.uint8)
            mask_b = (seg[:, :, 2] > 0.9).astype(np.uint8)
        elif color_str == "PURPLE":
            mask_r = (seg[:, :, 0] > 0.9).astype(np.uint8)
            mask_g = (seg[:, :, 1] < 0.1).astype(np.uint8)
            mask_b = (seg[:, :, 2] > 0.9).astype(np.uint8)
        return np.multiply(np.multiply(mask_r, mask_g), mask_b).astype(np.float32).copy()

    content_seg = np.array(Image.open(content_seg_path).convert("RGB").resize(content_shape, resample=Image.BILINEAR),
                           dtype=np.float32) / 255.0
    style_seg = np.array(Image.open(style_seg_path).convert("RGB").resize(style_shape, resample=Image.BILINEAR),
                         dtype=np.float32) / 255.0
    color_content_masks = []
    color_style_masks = []
    numOfColor = len(color_codes)
    for i in range(numOfColor):
        # color_content_masks.append(np.expand_dims(np.expand_dims(_extract_mask(content_seg, color_codes[i]), axis=0), axis=-1))
        # color_style_masks.append(np.expand_dims(np.expand_dims(_extract_mask(style_seg, color_codes[i]), axis=0), axis=-1))
        color_content_masks.append(_extract_mask(content_seg, color_codes[i]))
        color_style_masks.append(_extract_mask(style_seg, color_codes[i]))
    return color_content_masks, color_style_masks

def get_style_model_and_loss(content_seg_path,
                             style_seg_path,
                             style_img,
                             content_img,
                             loss_list,
                             cnn=vgg,
                             style_weight=800,
                             content_weight=3,
                             content_layers=content_layers_default,
                             style_layers=style_layers_default):
    content_loss_list = []
    style_loss_list = []
    content_masks, style_masks = load_seg(content_seg_path, style_seg_path, [content_img.shape[2], content_img.shape[3]],[style_img.shape[2], style_img.shape[3]])

    content_mask_test = torch.Tensor(content_masks).clone()
    style_mask_test = torch.Tensor(style_masks).clone()
    length_of_masks = len(content_masks)

    content_seg_height = np.array(content_masks).shape[1]
    content_seg_width = np.array(content_masks).shape[2]
    style_seg_height = np.array(style_masks).shape[1]
    style_seg_width = np.array(style_masks).shape[2]

    # test_image = np.zeros((content_seg_height, content_seg_width))
    # for k in range(content_seg_height):
    #     for l in range(content_seg_width):
    #         if content_masks[0][0][k][l][0] != 0:
    #             test_image[k][l] = 255
    #             print("content_masks[0][0][k][l][0]:  ", content_masks[0][0][k][l][0])
    # cv2.imwrite("test.png", test_image)
    # cv2.waitKey(0)

    kernel = [[1,1,1],[1,1,1],[1,1,1]]
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    weight = nn.Parameter(data = kernel,requires_grad = False)


    model = nn.Sequential()
    model = model.cuda()
    gram = loss.Gram()
    gram = gram.cuda()

    i = 1
    for layer in cnn:
        if isinstance(layer, nn.Conv2d):
            name = 'conv_' + str(i)
            model.add_module(name, layer)
            if name in content_layers_default:
                target = model(content_img)
                content_loss = loss.Content_Loss(target, content_weight)
                model.add_module('content_loss_' + str(i), content_loss)
                content_loss_list.append(content_loss)

            if name in style_layers_default:
                target = model(style_img)
                # num_channels = target.shape[1]
                # for mask_index in range(length_of_masks):
                #     for channel_index in range(num_channels):
                #         target[0][channel_index] = target[0][channel_index].mul(torch.Tensor(style_masks[mask_index]).cuda())
                #target = gram(target)

                new_content_masks = torch.ByteTensor(torch.Tensor(content_masks) > 0.6).float().cuda()
                new_style_masks = torch.ByteTensor(torch.Tensor(style_masks) > 0.6).float().cuda()
                # for index in range(target.shape[1]):
                #     new_target[0][index] = torch.mul(target[0][index], new_style_masks[semantic_index])

                style_loss = loss.Style_Loss(target, style_weight,new_content_masks, new_style_masks,loss_list)
                model.add_module('style_loss_' + str(i), style_loss)
                style_loss_list.append(style_loss)
            # for ii in range(length_of_masks):
            #     #conv_mask = nn.Conv2d(1,1,3,1,1,bias=False)
            #     style_masks[ii] = F.conv2d(torch.Tensor(style_masks[ii]).permute(0,3,1,2),weight = weight,padding = 1)
            #     content_masks[ii] = F.conv2d(torch.Tensor(content_masks[ii]).permute(0,3,1,2),weight = weight,padding = 1)
            #     style_masks[ii] = style_masks[ii].permute(0,2,3,1).detach().numpy()
            #     content_masks[ii] = content_masks[ii].permute(0,2,3,1).detach().numpy()



            i += 1
        if isinstance(layer, nn.MaxPool2d):
            name = 'pool_' + str(i)
            model.add_module(name, layer)

            content_seg_width, content_seg_height = int(math.ceil(content_seg_width / 2)), int(math.ceil(content_seg_height / 2))
            style_seg_width, style_seg_height = int(math.ceil(style_seg_width / 2)), int(math.ceil(style_seg_height / 2))
            for jj in range(length_of_masks):
                # new_content_masks = np.resize(content_masks[jj],(1,content_seg_height, content_seg_width,1)).copy()
                # new_style_masks = np.resize(style_masks[jj],(1,style_seg_height, style_seg_width,1)).copy()
                # content_masks.pop(jj)
                # content_masks.insert(jj,new_content_masks)
                # style_masks.pop(jj)
                # style_masks.insert(jj, new_style_masks)
                new_content_masks = cv2.resize(content_masks[jj], (content_seg_height, content_seg_width))
                new_style_masks = cv2.resize(style_masks[jj], (style_seg_height, style_seg_width))
                content_masks.pop(jj)
                content_masks.insert(jj,new_content_masks)
                style_masks.pop(jj)
                style_masks.insert(jj, new_style_masks)

        if isinstance(layer, nn.ReLU):
            name = 'relu' + str(i)
            model.add_module(name, layer)


    return model, style_loss_list, content_loss_list, content_mask_test, style_mask_test
