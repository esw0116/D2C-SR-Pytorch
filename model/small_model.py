'''

MegEngine is Licensed under the Apache License, Version 2.0 (the "License")

Copyright (c) Megvii Inc. All rights reserved.
Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

------------------------------------------------------------------------------
Part of the following code in this file refs to https://github.com/yulunzhang/RCAN
BSD 3-Clause License
Copyright (c) Soumith Chintala 2016,
All rights reserved.
--------------------------------------------------------------------------------

'''


import os, sys
sys.path.append(os.path.abspath('.'))
import torch
import torch.nn as nn
import model.model_utils as model_utils
from torch.nn import functional as F


class D_Net(nn.Module):
    def __init__(self, args):
        super(D_Net, self).__init__()
        if isinstance(args, dict):
            scale = args['scale']
        else:
            scale = args.scale
        self.scale = scale

        self.cand1 = nn.Sequential(
            model_utils.BasicConv(1, 64, 2, stride=1, padding=0, relu=True),
            model_utils.BasicConv(64, 64, 1, stride=1, padding=0, relu=True),
            model_utils.BasicConv(64, 64, 1, stride=1, padding=0, relu=True),
            model_utils.BasicConv(64, 64, 1, stride=1, padding=0, relu=True),
            model_utils.BasicConv(64, 64, 1, stride=1, padding=0, relu=True),
            model_utils.BasicConv(64, scale*scale, 1, stride=1, padding=0, relu=False),
            nn.PixelShuffle(scale))

        self.cand2 = nn.Sequential(
            model_utils.BasicConv(1, 64, 2, stride=1, padding=0, relu=True),
            model_utils.BasicConv(64, 64, 1, stride=1, padding=0, relu=True),
            model_utils.BasicConv(64, 64, 1, stride=1, padding=0, relu=True),
            model_utils.BasicConv(64, 64, 1, stride=1, padding=0, relu=True),
            model_utils.BasicConv(64, 64, 1, stride=1, padding=0, relu=True),
            model_utils.BasicConv(64, scale*scale, 1, stride=1, padding=0, relu=False),
            nn.PixelShuffle(scale))
        
        self.cand3 = nn.Sequential(
            model_utils.BasicConv(1, 64, 2, stride=1, padding=0, relu=True),
            model_utils.BasicConv(64, 64, 1, stride=1, padding=0, relu=True),
            model_utils.BasicConv(64, 64, 1, stride=1, padding=0, relu=True),
            model_utils.BasicConv(64, 64, 1, stride=1, padding=0, relu=True),
            model_utils.BasicConv(64, 64, 1, stride=1, padding=0, relu=True),
            model_utils.BasicConv(64, scale*scale, 1, stride=1, padding=0, relu=False),
            nn.PixelShuffle(scale))
                
        self.cand4 = nn.Sequential(
            model_utils.BasicConv(1, 64, 2, stride=1, padding=0, relu=True),
            model_utils.BasicConv(64, 64, 1, stride=1, padding=0, relu=True),
            model_utils.BasicConv(64, 64, 1, stride=1, padding=0, relu=True),
            model_utils.BasicConv(64, 64, 1, stride=1, padding=0, relu=True),
            model_utils.BasicConv(64, 64, 1, stride=1, padding=0, relu=True),
            model_utils.BasicConv(64, scale*scale, 1, stride=1, padding=0, relu=False),
            nn.PixelShuffle(scale))

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b*c, 1, h, w)

        res1 = self.cand1(x)
        res1 = res1.reshape(b, c, self.scale*(h-1), self.scale*(w-1))
        res2 = self.cand2(x)
        res2 = res2.reshape(b, c, self.scale*(h-1), self.scale*(w-1))
        res3 = self.cand3(x)
        res3 = res3.reshape(b, c, self.scale*(h-1), self.scale*(w-1))
        res4 = self.cand4(x)
        res4 = res4.reshape(b, c, self.scale*(h-1), self.scale*(w-1))

        return res1, res2, res3, res4

class C_Net(nn.Module):
    def __init__(self, details=False):
        super(C_Net, self).__init__()
        self.fusion_first = nn.Sequential(
            model_utils.BasicConv(4, 64, 1, stride=1, padding=0, relu=True),
            model_utils.BasicConv(64, 64, 1, stride=1, padding=0, relu=True),
            model_utils.BasicConv(64, 1, 1, stride=1, padding=0, relu=True))
        self.fusion_final = nn.Sequential(
            model_utils.BasicConv(3, 64, 1, stride=1, padding=0, relu=True),
            model_utils.BasicConv(64, 64, 1, stride=1, padding=0, relu=True),
            model_utils.BasicConv(64, 4, 1, stride=1, padding=0, relu=True))
        self.details = details

        gray_coeffs = [65.738, 129.057, 25.064]
        gray_coeffs_t = torch.Tensor(gray_coeffs).to('cuda')
        self.convert = gray_coeffs_t.reshape(1, 3, 1, 1) / 256

    def forward(self, x1,x2,x3,x4):
        x_r = torch.stack((x1[:, 0], x2[:, 0], x3[:, 0], x4[:, 0]), dim=1)
        x_g = torch.stack((x1[:, 1], x2[:, 1], x3[:, 1], x4[:, 1]), dim=1)
        x_b = torch.stack((x1[:, 2], x2[:, 2], x3[:, 2], x4[:, 2]), dim=1)
        x_r, x_g, x_b = self.fusion_first(x_r), self.fusion_first(x_g), self.fusion_first(x_b)
        cat_out = torch.cat((x_r,x_g,x_b), dim=1)
        mask = self.fusion_final(cat_out)
        mask = F.softmax(mask, dim=1)
        mask1 = mask[:,0:1,...]
        mask2 = mask[:,1:2, ...]
        mask3 = mask[:,2:3, ...]
        mask4 = mask[:,3:, ...]
        fusion = x1*mask1 + x2*mask2 + x3*mask3 + x4*mask4
        if self.details:
            return fusion, mask
        return fusion


if __name__ == '__main__':
    from config.config import args
    model_D=D_Net(args)
    print(model_D)
