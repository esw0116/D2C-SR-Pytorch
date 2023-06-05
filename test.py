'''

MegEngine is Licensed under the Apache License, Version 2.0 (the "License")

Copyright (c) Megvii Inc. All rights reserved.
Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

'''

import os, sys
sys.path.append(os.path.abspath('.'))
import imageio
from model.model import D_Net, C_Net
import torch
import torch.utils.data as data
import torch.distributed as dist
from torch.nn import functional as F

from dataset.dataset import *
from metrics.PSNR import cal_psnr
from utils import eval_metrics
from tqdm import tqdm

from config.config_test import args
import warnings


warnings.filterwarnings("ignore")


def main():
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # launch processes
    worker(args)


def worker(args):
    device = 'cuda'
    world_size = 1 # dist.get_world_size()
    # create dataset
    valid_dataloader, len_val = create_dataset(args)

    # create model
    model_D = D_Net(args)
    model_C = C_Net()

    with open(args.checkpoint, "rb") as f:
        state = torch.load(f)
    print("model path: %s"%(args.checkpoint))
    model_D.load_state_dict(state["state_dict_D"])
    model_C.load_state_dict(state["state_dict_C"])

    model_D, model_C = model_D.to(device), model_C.to(device)

    def valid_step(image, label, filename):
        with torch.no_grad():
            out11, out12, out21, out22 = model_D(image)
            sr = model_C(out11, out12, out21, out22)
        sr = sr.cpu(); label = label.cpu()
        if args.save_dir is not None:
            sr_np = sr.numpy()
            sr_np = sr_np[0].transpose(1,2,0)
            sr_np = sr_np[:,:, [2,1,0]]
            sr_np = np.clip(sr_np*255, 0, 255).round().astype(np.uint8)
            imageio.imwrite(os.path.join(args.save_dir, filename[0].split('/')[-1]), sr_np)

        _, ssim_it = eval_metrics(sr, label)
        psnr_it = cal_psnr(sr, label, args.scale)
        return psnr_it, ssim_it.item()

    model_D.eval()
    model_C.eval()
    psnr_v, ssim_v = valid(valid_step, valid_dataloader, len_val)
    print("PSNR [\033[1;31m{:.2f}\033[0m]  SSIM [\033[1;31m{:.3f}\033[0m] ".format(psnr_v, ssim_v))


def valid(func, data_queue, len_val):
    psnr_v = 0.
    ssim_v = 0.
    device = 'cuda'

    for step, (image, label, filename) in enumerate(tqdm(data_queue)):
        image, label = image.to(device), label.to(device)
        psnr_it, ssim_it = func(image, label, filename)
        psnr_v += psnr_it
        ssim_v += ssim_it
    test_num = step + 1
    psnr_v /= test_num
    ssim_v /= test_num
    assert test_num == len_val
    return psnr_v, ssim_v


def create_dataset(args):
    val_list_path = args.val_list_path
    test_list = open(val_list_path, 'r').readlines()
    len_val = len(test_list)
    valid_dataset = TestDataset(test_list)

    valid_sampler = data.SequentialSampler(valid_dataset)
    valid_dataloader = data.DataLoader(valid_dataset, sampler=valid_sampler, num_workers=args.workers, batch_size=1, drop_last=False)
    return valid_dataloader,len_val


if __name__ == "__main__":
    main()
