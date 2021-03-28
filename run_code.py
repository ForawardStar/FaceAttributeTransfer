import torch.nn as nn
import torch.optim as optim
import torch
import torchvision.transforms as transforms
import PIL.Image as Image
import cv2
import numpy as np
import torch.nn.functional as F
from torch.autograd import Function, Variable
import itertools
import dlib


from build_model import get_style_model_and_loss
from load_img import load_img

# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net,self).__init__()
#         self.hidden = torch.nn.Linear(1536*512,200)
#         self.out = torch.nn.Linear(200,1536*512)
#
#     def forward(self,x):
#         tmp_x = x.reshape(-1,1536*512)
#         tmp_x = F.relu(self.hidden(tmp_x))
#         res_x = self.out(tmp_x)
#         res_x = res_x.reshape(x.shape)
#         return res_x

class myloss(nn.Module):
    def __init__(self):
        super(myloss,self).__init__()
    def forward(self,previous,current):
        #out = torch.abs(torch.mean(previous[0][0]) - torch.mean(current[0][0])) + torch.abs(torch.mean(previous[0][1]) - torch.mean(current[0][1])) + torch.abs(torch.mean(previous[0][2]) - torch.mean(current[0][2]))
        # meanpre0 = torch.mean(previous[0][0])
        # meanpre1 = torch.mean(previous[0][1])
        # meanpre2 = torch.mean(previous[0][2])
        # meancurr0 = torch.mean(current[0][0])
        # meancurr1 = torch.mean(current[0][1])
        # meancurr2 = torch.mean(current[0][2])
        #
        # stdpre0 = torch.std(previous[0][0])
        # stdpre1 = torch.std(previous[0][1])
        # stdpre2 = torch.std(previous[0][2])
        # stdcurr0 = torch.std(current[0][0])
        # stdcurr1 = torch.std(current[0][1])
        # stdcurr2 = torch.std(current[0][2])
        #
        # c1 = (0.001 / 1)**2
        # c2 = (0.003 / 1) ** 2
        #
        # out1 = (2*meanpre0*meancurr0+c1)/(meanpre0*meanpre0+meancurr0*meancurr0+c1) + (2*meanpre1*meancurr1+c1)/(meanpre1*meanpre1+meancurr1*meancurr1+c1) + (2*meanpre2*meancurr2+c1)/(meanpre2*meanpre2+meancurr2*meancurr2+c1)
        # out2 = (2*stdpre0*stdcurr0 + c2)/(stdpre0*stdpre0 + stdcurr0*stdcurr0 + c2) + (2*stdpre1*stdcurr1 + c2)/(stdpre1*stdpre1 + stdcurr1*stdcurr1 + c2) + (2*stdpre2*stdcurr2 + c2)/(stdpre2*stdpre2 + stdcurr2*stdcurr2 + c2)
        # out = out1 * out2

        out = nn.MSELoss()(previous[0],current[0])

        return out

def key_points(img):
    img = cv2.cvtColor(np.asarray(img),cv2.COLOR_BGR2RGB)
    points_key = []
    PREDICITOR_PATH = 'shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICITOR_PATH)

    rects = detector(img,1)

    for i in range(len(rects)):
        landmarks = np.matrix([[p.x,p.y] for p in predictor(img,rects[i]).parts()])
        img = img.copy()
        for idx,point in enumerate(landmarks):
            pos = (point[0,0],point[0,1])
            points_key.append(pos)
            cv2.circle(img,pos,2,(255,0,0),-1)
    cv2.imwrite("test.png",img)
    return torch.Tensor(points_key).cuda()

def grid_sample(input, grid, canvas = None):
    output = F.grid_sample(input, grid)
    if canvas is None:
        return output
    else:
        input_mask = Variable(input.data.new(input.size()).fill_(1))
        output_mask = F.grid_sample(input_mask, grid)
        padded_output = output * output_mask + canvas * (1 - output_mask)
        return padded_output

def compute_partial_repr(input_points, control_points):
    N = input_points.size(0)
    M = control_points.size(0)
    pairwise_diff = input_points.view(N, 1, 2) - control_points.view(1, M, 2)
    # original implementation, very slow
    # pairwise_dist = torch.sum(pairwise_diff ** 2, dim = 2) # square of distance
    pairwise_diff_square = pairwise_diff * pairwise_diff
    pairwise_dist = pairwise_diff_square[:, :, 0] + pairwise_diff_square[:, :, 1]
    repr_matrix = 0.5 * pairwise_dist * torch.log(pairwise_dist)
    # fix numerical error for 0 * log(0), substitute all nan with 0
    mask = repr_matrix != repr_matrix
    repr_matrix.masked_fill_(mask, 0)
    return repr_matrix

class TPSGridGen(nn.Module):
    def __init__(self, target_height, target_width, target_control_points):
        super(TPSGridGen, self).__init__()
        assert target_control_points.ndimension() == 2
        assert target_control_points.size(1) == 2
        N = target_control_points.size(0)
        self.num_points = N
        target_control_points = target_control_points.float()

        # create padded kernel matrix
        forward_kernel = torch.zeros(N + 3, N + 3)
        target_control_partial_repr = compute_partial_repr(target_control_points, target_control_points)
        forward_kernel[:N, :N].copy_(target_control_partial_repr)
        forward_kernel[:N, -3].fill_(1)
        forward_kernel[-3, :N].fill_(1)
        forward_kernel[:N, -2:].copy_(target_control_points)
        forward_kernel[-2:, :N].copy_(target_control_points.transpose(0, 1))
        # compute inverse matrix
        inverse_kernel = torch.inverse(forward_kernel)

        # create target cordinate matrix
        HW = target_height * target_width
        target_coordinate = list(itertools.product(range(target_height), range(target_width)))
        target_coordinate = torch.Tensor(target_coordinate) # HW x 2
        Y, X = target_coordinate.split(1, dim = 1)
        Y = Y * 2 / (target_height - 1) - 1
        X = X * 2 / (target_width - 1) - 1
        target_coordinate = torch.cat([X, Y], dim = 1) # convert from (y, x) to (x, y)
        target_coordinate = target_coordinate.cuda()

        target_coordinate_partial_repr = compute_partial_repr(target_coordinate, target_control_points)
        target_coordinate_repr = torch.cat([
            target_coordinate_partial_repr, torch.ones(HW, 1).cuda(), target_coordinate
        ], dim = 1)

        # register precomputed matrices
        self.register_buffer('inverse_kernel', inverse_kernel)
        self.register_buffer('padding_matrix', torch.zeros(3, 2))
        self.register_buffer('target_coordinate_repr', target_coordinate_repr)

    def forward(self, source_control_points):
        assert source_control_points.ndimension() == 2
        assert source_control_points.size(0) == self.num_points
        assert source_control_points.size(1) == 2

        Y = torch.cat([source_control_points, Variable(self.padding_matrix.expand(3, 2))], 0)

        mapping_matrix = torch.matmul(Variable(self.inverse_kernel), Y)
        source_coordinate = torch.matmul(Variable(self.target_coordinate_repr), mapping_matrix)

        return source_coordinate

class CNN(nn.Module):
    def __init__(self, num_output):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(312500, 50)
        self.fc2 = nn.Linear(50, num_output)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        numnerul = x.shape[1]*x.shape[2]*x.shape[3]
        x = x.view(-1, numnerul)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

class ClsNet(nn.Module):

    def __init__(self):
        super(ClsNet, self).__init__()
        self.cnn = CNN(10)

    def forward(self, x):
        return F.log_softmax(self.cnn(x))

class BoundedGridLocNet(nn.Module):

    def __init__(self, grid_height, grid_width, target_control_points):
        super(BoundedGridLocNet, self).__init__()
        self.cnn = CNN(grid_height * grid_width * 2)

        bias = torch.from_numpy(np.arctanh(target_control_points.numpy()))
        bias = bias.view(-1)
        self.cnn.fc2.bias.data.copy_(bias)
        self.cnn.fc2.weight.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)
        points = F.tanh(self.cnn(x))
        return points.view(batch_size, -1, 2)

class UnBoundedGridLocNet(nn.Module):

    def __init__(self, grid_number, target_control_points):
        super(UnBoundedGridLocNet, self).__init__()
        self.cnn = CNN(grid_number * 2)

        bias = target_control_points.view(-1)
        self.cnn.fc2.bias.data.copy_(bias)
        self.cnn.fc2.weight.data.zero_()
    def forward(self, x):
        points = self.cnn(x)
        return points.view(1, -1, 2)

class STNClsNet(nn.Module):
    def __init__(self, target_control_points, choose,image_height,image_width,span_range_height = 0.9,span_range_width = 0.9):
        super(STNClsNet, self).__init__()

        self.image_height = image_height
        self.image_width = image_width

        # r1 = span_range_height
        # r2 = span_range_width
        # assert r1 < 1 and r2 < 1 # if >= 1, arctanh will cause error in BoundedGridLocNet
        # target_control_points = torch.Tensor(list(itertools.product(
        #     np.arange(-r1, r1 + 0.00001, 2.0  * r1 / (grid_height - 1)),
        #     np.arange(-r2, r2 + 0.00001, 2.0  * r2 / (grid_width - 1)),
        # )))
        # Y, X = target_control_points.split(1, dim = 1)
        # target_control_points = torch.cat([X, Y], dim = 1)
        # print("target_control_points type: ",type(target_control_points))
        # print("target_control_points shape: ",target_control_points.shape)

        # GridLocNet = {
        #     'unbounded_stn': UnBoundedGridLocNet,
        #     'bounded_stn': BoundedGridLocNet,
        # }[choose]

        grid_number = len(target_control_points)
        X, Y = target_control_points.split(1, dim=1)
        Y = Y * 2 / (image_height - 1) - 1
        X = X * 2 / (image_width - 1) - 1
        target_control_points = torch.cat([X, Y], dim=1)  # convert from (y, x) to (x, y)


        self.loc_net = UnBoundedGridLocNet(grid_number, target_control_points)

        self.tps = TPSGridGen(image_height, image_width, target_control_points)

        self.cls_net = ClsNet()

    def forward(self, x,source_control_points):
        #source_control_points = self.loc_net(x)
        X, Y = source_control_points.split(1, dim=1)
        Y = Y * 2 / (self.image_height - 1) - 1
        X = X * 2 / (self.image_width - 1) - 1
        source_control_points = torch.cat([X, Y], dim=1)  # convert from (y, x) to (x, y)
        source_coordinate = self.tps(source_control_points)
        grid = source_coordinate.view(1, self.image_height, self.image_width, 2)
        transformed_x = grid_sample(x, grid)
        return transformed_x.clone()


def get_input_param_optimier(input_img):
    """
    input_img is a Variable
    """
    input_param = nn.Parameter(input_img.data)
    optimizer = optim.LBFGS([input_param])
    return input_param, optimizer

def run_style_transfer(content_seg_path, style_seg_path,content_img, style_img, input_img, num_epoches=1500):
    loss_list = [0,3,4,5]
    background_list = [1,2]

    print('Building the style transfer model..')
    content_ketpoints = key_points(transforms.ToPILImage()(content_img.cpu().squeeze(0)))
    style_ketpoints = key_points(transforms.ToPILImage()(style_img.cpu().squeeze(0)))
    print("content_ketpoints:  ",content_ketpoints)
    print("style_ketpoints:  ", style_ketpoints)

    model, style_loss_list, content_loss_list,content_mask,style_mask = get_style_model_and_loss(content_seg_path, style_seg_path, style_img, content_img, loss_list)

    input_param, optimizer = get_input_param_optimier(input_img)
    RGBoptimizer = optim.LBFGS([input_param])

    warpmodel = STNClsNet(content_ketpoints,'unbounded_stn',512,512).cuda()
    RGBloss = myloss()

    print('Opimizing...')
    epoch = [0]

    foreground_mask = torch.zeros(style_mask[0].shape)
    background_mask = torch.zeros(content_mask[0].shape)
    for index_fore in loss_list:
        foreground_mask = foreground_mask + style_mask[index_fore]
    for index_back in background_list:
        background_mask = background_mask + content_mask[index_back]
    input_param.data = input_param.data.mul(foreground_mask.cuda())
    input_param.data = warpmodel(input_param, style_ketpoints)

    tmp_input_param = input_param.clone()
    tmp_input_param = tmp_input_param.cpu()
    tmp_input_param = tmp_input_param.squeeze(0)
    tmp_input_param = transforms.ToPILImage()(tmp_input_param)
    tmp_input_param.save("warp_result.png")

    input_rows = input_param.shape[2]
    input_cols = input_param.shape[3]
    for indi in range(input_rows):
        for indj in range(input_cols):
            if background_mask[indi][indj] != 0 or (input_param[0][0][indi][indj] == 0 and input_param[0][1][indi][indj] == 0 and input_param[0][2][indi][indj] == 0):
                #print("coordinate:  ({},{})".format(indi, indj))
                input_param.data[0][0][indi][indj] = content_img[0][0][indi][indj]
                input_param.data[0][1][indi][indj] = content_img[0][1][indi][indj]
                input_param.data[0][2][indi][indj] = content_img[0][2][indi][indj]
            # if content_mask[0][indi][indj] == 1 and input_param[0][0][indi][indj] == 0 and input_param[0][1][indi][indj] == 0 and input_param[0][2][indi][indj] == 0:
            #     input_param.data[0][0][indi][indj] = stylemean0
            #     input_param.data[0][1][indi][indj] = stylemean1
            #     input_param.data[0][2][indi][indj] = stylemean2

    tmp_input_param = input_param.clone()
    tmp_input_param = tmp_input_param.cpu()
    tmp_input_param = tmp_input_param.squeeze(0)
    tmp_input_param = transforms.ToPILImage()(tmp_input_param)
    tmp_input_param.save("fusion_result.png")

    count = 0
    step = 1
    record_input = content_img.data.clone()
    while epoch[0] <= num_epoches:
        def closure():

            input_param.data.clamp_(0, 1)

            model(input_param)

            style_score = 0
            content_score = 0

            optimizer.zero_grad()
            for sl in style_loss_list:
                style_score += sl.backward()
            for cl in content_loss_list:
                content_score += cl.backward()

            epoch[0] += 1

            # if style_score.data[0] > record[0] and content_score.data[0] > record[1]:
            #     print("final epoch:  ",epoch[0])
            #     tmp_input_param = input_param.cpu()
            #     tmp_input_param = tmp_input_param.squeeze(0)
            #     tmp_input_param = transforms.ToPILImage()(tmp_input_param)
            #     tmp_input_param.save("newresult_step{}.png".format(epoch[0]))
            #     epoch[0] = num_epoches + 1
            #
            # record[0] = style_score.data[0].item()
            # record[1] = content_score.data[0].item()

            if epoch[0] % 30 == 0:
                print('run {}'.format(epoch))
                print('Style Loss: {:.4f} Content Loss: {:.4f}'.format(
                    style_score.data[0], content_score.data[0]))
                print()
                tmp_input_param = input_param.cpu()
                tmp_input_param = tmp_input_param.squeeze(0)
                tmp_input_param = transforms.ToPILImage()(tmp_input_param)
                tmp_input_param.save("result{}.png".format(epoch[0]))

            return content_score + style_score

        def closure2():
            input_param.data.clamp_(0, 1)

            rgbloss = RGBloss(record_input.mul(background_mask.cuda()), input_param.mul(background_mask.cuda()))
            RGBoptimizer.zero_grad()
            rgbloss.backward()

            epoch[0] += 1
            if epoch[0] % 30 == 0:
                print('run {}'.format(epoch))
                print("rgbloss:  ",rgbloss)
                tmp_input_param = input_param.cpu()
                tmp_input_param = tmp_input_param.squeeze(0)
                tmp_input_param = transforms.ToPILImage()(tmp_input_param)
                tmp_input_param.save("result{}.png".format(epoch[0]))

                # print('Style Loss: {:.4f} Content Loss: {:.4f}'.format(
                #     style_score.data[0], content_score.data[0]))
                # print()
                # tmp_input_param = input_param.cpu()
                # tmp_input_param = tmp_input_param.squeeze(0)
                # tmp_input_param = transforms.ToPILImage()(tmp_input_param)
                # tmp_input_param.save("result{}.png".format(epoch[0]))

            return rgbloss

        # input_keypoints =  key_points(transforms.ToPILImage()(input_param.data.cpu().squeeze(0)))
        #record_input = input_param.data.clone()
        optimizer.step(closure)
        input_param.data.clamp_(0, 1)

        #if count >= step:
            #input_param.data = warpmodel(input_param, input_keypoints)
        RGBoptimizer.step(closure2)
            #step = step + 1
            #count = 0
        #count = count + 1

    return input_param.data

content_img = load_img("in11.png").cuda()
style_img = load_img("tar11.png").cuda()

#input_img = load_img("warp_result.png").cuda()
#input_img = torch.randn(content_img.shape).cuda()
input_img = style_img.clone()
content_seg_path = "in11_seg.png"
style_seg_path = "tar11_seg.png"

input_param = run_style_transfer(content_seg_path, style_seg_path, content_img, style_img, input_img)

input_param = input_param.cpu()
input_param = input_param.squeeze(0)
input_param = transforms.ToPILImage()(input_param)
input_param.save("result.png")