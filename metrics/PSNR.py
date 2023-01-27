'''

MegEngine is Licensed under the Apache License, Version 2.0 (the "License")

Copyright (c) Megvii Inc. All rights reserved.
Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

'''

import os,sys
import math
sys.path.append(os.path.abspath('.'))
import torch
from torch.nn import functional as F


def cal_psnr(sr, hr, shave):
    if not sr.shape == hr.shape:
        raise ValueError('Input images must have the same dimensions.')
    sr = sr.squeeze(0).clip(0, 1)
    hr = hr.squeeze(0).clip(0, 1)
    diff = (sr - hr)
    gray_coeffs = [65.738, 129.057, 25.064]
    gray_coeffs_t = torch.Tensor(gray_coeffs)
    convert = gray_coeffs_t.reshape(3, 1, 1) / 256
    diff = diff.mul(convert)
    diff = diff.sum(dim=0, keepdims=True)
    valid = diff[..., shave:-shave, shave:-shave]
    mse = torch.mean(torch.pow(valid,2))
    return -10 * math.log10(mse)


if __name__ == '__main__':
    a = torch.rand(3, 10, 10)
    b = torch.rand(3, 10, 10)
    print(cal_psnr(a,b,4))


