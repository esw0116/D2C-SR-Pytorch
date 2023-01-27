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
import megengine.module as nn
import model.model_utils as model_utils
import megengine as mge
import megengine.functional as F

class D_Net(nn.Module):
    def __init__(self, args):
        super(D_Net, self).__init__()
        if isinstance(args, dict):
            scale = args['scale']
        else:
            scale = args.scale
        self.scale = scale
        # self.sub_mean = model_utils.MeanShift(rgb_range)
        # self.add_mean = model_utils.MeanShift(rgb_range, sign=1)
        
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
    def __init__(self):
        super(C_Net, self).__init__()
        self.fusion_final1 = nn.Sequential(
            model_utils.BasicConv(4, 64, 1, stride=1, padding=0, relu=True),
            model_utils.BasicConv(64, 64, 1, stride=1, padding=0, relu=True),
            model_utils.BasicConv(64, 64, 1, stride=1, padding=0, relu=True),
            model_utils.BasicConv(64, 4, 1, stride=1, padding=0, relu=True))
        
        self.fusion_final2 = nn.Sequential(
            model_utils.BasicConv(4, 64, 1, stride=1, padding=0, relu=True),
            model_utils.BasicConv(64, 64, 1, stride=1, padding=0, relu=True),
            model_utils.BasicConv(64, 64, 1, stride=1, padding=0, relu=True),
            model_utils.BasicConv(64, 4, 1, stride=1, padding=0, relu=True))
        
        self.fusion_final3 = nn.Sequential(
            model_utils.BasicConv(4, 64, 1, stride=1, padding=0, relu=True),
            model_utils.BasicConv(64, 64, 1, stride=1, padding=0, relu=True),
            model_utils.BasicConv(64, 64, 1, stride=1, padding=0, relu=True),
            model_utils.BasicConv(64, 4, 1, stride=1, padding=0, relu=True))

        self.fusion_final = nn.Sequential(
            model_utils.BasicConv(3, 64, 1, stride=1, padding=0, relu=True),
            model_utils.BasicConv(64, 64, 1, stride=1, padding=0, relu=True),
            model_utils.BasicConv(64, 64, 1, stride=1, padding=0, relu=True),
            model_utils.BasicConv(64, 3, 1, stride=1, padding=0, relu=True))

    def forward(self, x1,x2,x3,x4):
        stack_out=F.stack((x1,x2,x3,x4),axis=2)
        
        stack_out_r = stack_out[:,0] # B 4 H W
        mask_r = self.fusion_final1(stack_out_r) # B 4 H W
        mask_r = F.softmax(mask_r, axis=1)
        fusion_r = F.sum(stack_out_r * mask_r, axis=1)

        stack_out_g = stack_out[:,1] # B 4 H W
        mask_g = self.fusion_final2(stack_out_g) # B 4 H W
        mask_g = F.softmax(mask_g, axis=1)
        fusion_g = F.sum(stack_out_g * mask_g, axis=1)
        
        stack_out_b = stack_out[:,0] # B 4 H W
        mask_b = self.fusion_final3(stack_out_b) # B 4 H W
        mask_b = F.softmax(mask_b, axis=1)
        fusion_b = F.sum(stack_out_b * mask_b, axis=1)
        
        fusion = F.stack((fusion_r, fusion_g, fusion_b), axis=1)
        fusion = self.fusion_final(fusion)
        return fusion


class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class RCAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


if __name__ == '__main__':
    from config.config import args
    model_D=D_Net(args)
    print(model_D)
