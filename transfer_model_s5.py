
from importlib import import_module
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


exp_name = 'simple5'
UPSCALE = 4                  # upscaling factor
MODEL_PATH = "experiments/pytorch_{}_triploss_x4/checkpoint.ckpt".format(exp_name)  # Trained SR net params
SAMPLING_INTERVAL = 4        # N bit uniform sampling
SAVE_DIR = os.path.join(os.path.dirname(MODEL_PATH), "LUTs")

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


# from model.simple5_model import D_Net, C_Net
module = import_module('model.{}_model'.format(exp_name))
D_Net = getattr(module, 'D_Net')
C_Net = getattr(module, 'C_Net')

args = {}
args['scale'] = UPSCALE
model_D = D_Net(args)
# model_D = D_Net(scale=UPSCALE)
model_C = C_Net()

lm = torch.load('{}'.format(MODEL_PATH))
print("model path: %s"%(MODEL_PATH))
model_D.load_state_dict(lm["state_dict_D"])
model_C.load_state_dict(lm["state_dict_C"])

model_D = model_D.cuda()

### Extract input-output pairs
model_D.eval()
model_C.eval()


### Extract input-output pairs
with torch.no_grad():
    # 1D input
    base = torch.arange(0, 257, 2**SAMPLING_INTERVAL)   # 0-256
    base[-1] -= 1
    L = base.size(0)

    # 2D input
    first = base.cuda().unsqueeze(1).repeat(1, L).reshape(-1)  # 256*256   0 0 0...    |1 1 1...     |...|255 255 255...
    second = base.cuda().repeat(L)                             # 256*256   0 1 2 .. 255|0 1 2 ... 255|...|0 1 2 ... 255
    onebytwo = torch.stack([first, second], 1)  # [256*256, 2]

    # 3D input
    third = base.cuda().unsqueeze(1).repeat(1, L*L).reshape(-1) # 256*256*256   0 x65536|1 x65536|...|255 x65536
    onebytwo = onebytwo.repeat(L, 1)
    onebythree = torch.cat([third.unsqueeze(1), onebytwo], 1)    # [256*256*256, 3]

    # 4D input
    fourth = base.cuda().unsqueeze(1).repeat(1, L*L*L).reshape(-1) # 256*256*256*256   0 x16777216|1 x16777216|...|255 x16777216
    onebythree = onebythree.repeat(L, 1)
    onebyfourth = torch.cat([fourth.unsqueeze(1), onebythree], 1)    # [256*256*256*256, 4]

    # Rearange input: [N, 4] -> [N, C=1, H=2, W=2]
    input_tensor = onebyfourth.unsqueeze(1).unsqueeze(1).reshape(-1,1,2,2).float() / 255.0
    print("Input size: ", input_tensor.size())

    # Split input to not over GPU memory
    B = input_tensor.size(0) // 100

    for i in range(4):
        outputs = []
        for b in range(100):
            if b == 99:
                batch_output = model_D(input_tensor[b*B:])
            else:
                batch_output = model_D(input_tensor[b*B:(b+1)*B])

            results = torch.round(torch.clamp(batch_output[i], -1, 1)*127).cpu().data.numpy().astype(np.int8)
            outputs += [ results ]
        
        results = np.concatenate(outputs, 0)
        print("Resulting LUT size: ", results.shape)

        # np.save("Model_S_x{}_{}bit_int8".format(UPSCALE, SAMPLING_INTERVAL), results)
        np.save(os.path.join(SAVE_DIR, "Model_D_{}_x{}_{}bit".format(i, UPSCALE, SAMPLING_INTERVAL)), results)


### Extract input-output pairs
with torch.no_grad():
    # 1D input
    base = torch.arange(0, 257, 2**SAMPLING_INTERVAL)   # 0-256
    base[-1] -= 1
    L = base.size(0)

    # 2D input
    first = base.cuda().unsqueeze(1).repeat(1, L).reshape(-1)  # 256*256   0 0 0...    |1 1 1...     |...|255 255 255...
    second = base.cuda().repeat(L)                             # 256*256   0 1 2 .. 255|0 1 2 ... 255|...|0 1 2 ... 255
    onebytwo = torch.stack([first, second], 1)  # [256*256, 2]

    # 3D input
    third = base.cuda().unsqueeze(1).repeat(1, L*L).reshape(-1) # 256*256*256   0 x65536|1 x65536|...|255 x65536
    onebytwo = onebytwo.repeat(L, 1)
    onebythree = torch.cat([third.unsqueeze(1), onebytwo], 1)    # [256*256*256, 3]

    # 4D input
    fourth = base.cuda().unsqueeze(1).repeat(1, L*L*L).reshape(-1) # 256*256*256*256   0 x16777216|1 x16777216|...|255 x16777216
    onebythree = onebythree.repeat(L, 1)
    onebyfourth = torch.cat([fourth.unsqueeze(1), onebythree], 1)    # [256*256*256*256, 4]

    # Rearange input: [N, 4] -> [N, C=1, H=2, W=2]
    input_tensor = onebyfourth.unsqueeze(1).unsqueeze(1).reshape(-1,1,2,2).float() / 255.0
    print("Input size: ", input_tensor.size())

    # Split input to not over GPU memory
    B = input_tensor.size(0) // 100

    for i in range(4):
        outputs = []
        for b in range(100):
            if b == 99:
                batch_output = model_D(input_tensor[b*B:])
            else:
                batch_output = model_D(input_tensor[b*B:(b+1)*B])

            results = torch.round(torch.clamp(batch_output[i], -1, 1)*127).cpu().data.numpy().astype(np.int8)
            outputs += [ results ]
        
        results = np.concatenate(outputs, 0)
        print("Resulting LUT size: ", results.shape)

        # np.save("Model_S_x{}_{}bit_int8".format(UPSCALE, SAMPLING_INTERVAL), results)
        np.save(os.path.join(SAVE_DIR, "Model_D_{}_x{}_{}bit".format(i, UPSCALE, SAMPLING_INTERVAL)), results)


