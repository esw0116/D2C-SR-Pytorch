'''

MegEngine is Licensed under the Apache License, Version 2.0 (the "License")

Copyright (c) Megvii Inc. All rights reserved.
Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

'''

import os, sys
sys.path.append(os.path.abspath('.'))
import time
import logging
import numpy as np
from model.model import D_Net, C_Net
import torch
import torch.utils.data as data
import torch.distributed as dist
import torch.optim as optim
from torch.nn import functional as F

from dataset.dataset import *
from metrics.PSNR import cal_psnr
# from metrics.SSIM import *
from loss.divergence import trip_loss
from utils import eval_metrics

from config.config import args
import warnings

logger = logging.getLogger()
warnings.filterwarnings("ignore")

def main():
    ngpus_per_node = 1
    if args.ngpus:
        ngpus_per_node = args.ngpus
    save_dir = "./experiments/"
    args.ex_name = "%s_x%s" % (args.ex_id, args.scale)
    args.save = os.path.join(save_dir, args.ex_name)
    worker(args)


def worker(args):
    world_size = 1
    device = 'cuda'
    print("world size : %s" % world_size)
    os.makedirs(os.path.join(args.save), exist_ok=True)

    # create dataset
    train_dataloader, valid_dataloader, len_train, len_val = create_dataset(args)
    steps_per_epoch = len_train // args.batch_size
    print("steps_per_epoch : %s " % steps_per_epoch)
    train_queue = iter(train_dataloader)

    # create model
    model_D = D_Net(args)
    model_C = C_Net()

    if (args.load_checkpoint):
        with open(args.checkpoint, "rb") as f:
            state = torch.load(f)
        print("model path: %s  |  PSNR: [%s]" % (args.checkpoint, state["psnr"]))
        model_D.load_state_dict(state["state_dict_D"])
        model_C.load_state_dict(state["state_dict_C"])

    model_D, model_C = model_D.to(device), model_C.to(device)

    # define Optimizer
    opt_C = optim.Adam(model_C.parameters(), lr=args.lr, weight_decay=args.weight_decay,)


    def train_step_C(out11, out12, out21, out22, label):
        opt_C.zero_grad()
        fusion = model_C(out11, out12, out21, out22)
        loss_C = F.mse_loss(fusion, label)
        loss_C.backward()
        opt_C.step()

        return loss_C

        weights_before = copy.deepcopy(model_C.state_dict())
        inner_optimizer = optim.Adamax(model_C.parameters(), lr=cfg.inner_lr)

        # inner loop
        for _k in range(k):
            total_loss = 0
            output = model_C(out11, out12, out21, out22)
            inner_loss = F.mse_loss(output, label)

            inner_optimizer.zero_grad()
            inner_loss.backward()
            inner_optimizer.step()

        # Reptile - outer update
        outerstepsize = cfg.outer_lr
        weights_after = model_C.state_dict()
        model_C.load_state_dict({name:
            weights_before[name] + (weights_after[name] - weights_before[name]) * outerstepsize
            for name in weights_before})

        # calculate loss w/ updated model
        input0, input1, target = images[2], images[4], images[3]

        with torch.no_grad():
            output = model_C(input0, input1)
            loss = criterion(output, target)

    def valid_step(image, label):
        with torch.no_grad():
            out11, out12, out21, out22 = model_D(image)
            sr = model_C(out11, out12, out21, out22)
        sr = sr.cpu(); label = label.cpu()
        _, ssim_it = eval_metrics(sr, label)
        psnr_it = cal_psnr(sr, label, args.scale)

        return psnr_it, ssim_it

    # multi-step learning rate scheduler with warmup
    def adjust_learning_rate(step, opt):
        lr = args.lr * (args.gamma ** ((step / steps_per_epoch) // args.decay_epoch))
        for param_group in opt.param_groups:
            param_group["lr"] = lr
        return lr

    model_D.eval()
    # start training
    for step in range(0, int(args.epochs * steps_per_epoch)):
        lr_C = adjust_learning_rate(step, opt_C)
        t_step = time.time()

        try:
            image, label = next(train_queue)
        except StopIteration:
            train_queue = iter(train_dataloader)
            image, label = next(train_queue)
        
        image, label = image.to(device), label.to(device)

        with torch.no_grad():
            out11, out12, out21, out22 = model_D(image)
        
        loss_C = train_step_C(out11, out12, out21, out22, label)
        t_train = time.time() - t_step
        if step % args.print_freq == 0:
            logger.info("[{}]  Epoch {} Step {}\t Loss_C={:.5}  lr={:.5}\t times={:.2}s".format(
                    args.ex_name, step // steps_per_epoch, step, loss_C.item(), t_train))
            print("[{}]  Epoch {} Step {}\t Loss_C={:.5}  lr={:.5}\t times={:.2}s".format(
                    args.ex_name, step // steps_per_epoch, step, loss_C.item(), t_train))

        if ((step + 1) % (steps_per_epoch * args.val_freq) == 0):
            model_C.eval()
            
            psnr_v, ssim_v = valid(valid_step, valid_dataloader, len_val)
            model_C.train()

            model_dict = {
                "state_dict_D": model_D.to('cpu').state_dict(),
                "state_dict_C": model_C.to('cpu').state_dict(),
                }
            torch.save(model_dict, os.path.join(args.save, "checkpoint.ckpt"))
            logger.info(
                "[{}]  PSNR [\033[1;31m{:.2f}\033[0m]  SSIM [\033[1;31m{:.4f}\033[0m]".format(
                    args.ex_name, psnr_v, ssim_v))
            print("[{}]  PSNR [\033[1;31m{:.2f}\033[0m]  SSIM [\033[1;31m{:.4f}\033[0m]".format(
                    args.ex_name, psnr_v, ssim_v))


def valid(func, data_queue, len_val):
    psnr_v = 0.
    ssim_v = 0.
    device = 'cuda'

    for step, (image, label, _) in enumerate(data_queue):
        image, label = image.to(device), label.to(device)
        psnr_it, ssim_it = func(image, label)
        psnr_v += psnr_it
        ssim_v += ssim_it
    test_num = step + 1
    psnr_v /= test_num
    ssim_v /= test_num
    assert test_num == len_val
    return psnr_v, ssim_v.item()


def create_dataset(args):
    train_list_path = args.train_list_path
    val_list_path = args.val_list_path
    if args.debug:
        train_list = open(train_list_path, 'r').readlines()[:args.batch_size]
        test_list = open(val_list_path, 'r').readlines()[:1]
    else:
        train_list = open(train_list_path, 'r').readlines()
        test_list = open(val_list_path, 'r').readlines()

    assert not args.batch_size // args.ngpus == 0

    len_train = len(train_list)
    len_val = len(test_list)
    train_dataset = TrainDataset(train_list, args)
    valid_dataset = TestDataset(test_list)


    train_sampler = data.RandomSampler(train_dataset)
    train_dataloader = data.DataLoader(train_dataset, sampler=train_sampler, num_workers=args.workers, batch_size=args.batch_size // args.ngpus, drop_last=True)

    valid_sampler = data.SequentialSampler(valid_dataset)
    valid_dataloader = data.DataLoader(valid_dataset, sampler=valid_sampler, num_workers=args.workers, batch_size=1, drop_last=False)

    return train_dataloader, valid_dataloader, len_train, len_val


if __name__ == "__main__":
    main()
    # python train.py --ex_id pytorch --train_list_path dataset/RealSRTrain.txt --val_list_path dataset/RealSRTest.txt