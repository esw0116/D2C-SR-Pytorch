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

import torch
import torch.utils.data as data
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from dataset.dataset import *
from metrics.PSNR import cal_psnr
from loss.divergence import trip_loss
from utils import eval_metrics
from model.small_model import D_Net, C_Net

from config.config import args
import warnings

logger = logging.getLogger()
warnings.filterwarnings("ignore")


def main():
    save_dir = "./experiments/"
    args.ex_name = "%s_x%s" % (args.ex_id, args.scale)
    args.save = os.path.join(save_dir, args.ex_name)
    writer = SummaryWriter(log_dir=args.save)

    diver_w = 0.

    device = 'cuda'
    os.makedirs(os.path.join(args.save), exist_ok=True)
    logging.basicConfig(filename=os.path.join(args.save, "log.log"))

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
    opt_D = optim.Adam(model_D.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt_C = optim.Adam(model_C.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # train and valid
    def train_step_D(image, label):
        opt_D.zero_grad()
        out11_S1, out12_S1, out21_S1, out22_S1 = model_D(F.pad(image, pad=(0,1,0,1), mode='reflect'))
        out11_S2, out12_S2, out21_S2, out22_S2 = model_D(F.pad(torch.rot90(image, 1, [2,3]), pad=(0,1,0,1), mode='reflect'))
        out11_S2, out12_S2, out21_S2, out22_S2 = torch.rot90(out11_S2, 3, [2,3]), torch.rot90(out12_S2, 3, [2,3]), torch.rot90(out21_S2, 3, [2,3]), torch.rot90(out22_S2, 3, [2,3])
        out11_S3, out12_S3, out21_S3, out22_S3 = model_D(F.pad(torch.rot90(image, 2, [2,3]), pad=(0,1,0,1), mode='reflect'))
        out11_S3, out12_S3, out21_S3, out22_S3 = torch.rot90(out11_S3, 2, [2,3]), torch.rot90(out12_S3, 2, [2,3]), torch.rot90(out21_S3, 2, [2,3]), torch.rot90(out22_S3, 2, [2,3])
        out11_S4, out12_S4, out21_S4, out22_S4 = model_D(F.pad(torch.rot90(image, 3, [2,3]), pad=(0,1,0,1), mode='reflect'))
        out11_S4, out12_S4, out21_S4, out22_S4 = torch.rot90(out11_S4, 1, [2,3]), torch.rot90(out12_S4, 1, [2,3]), torch.rot90(out21_S4, 1, [2,3]), torch.rot90(out22_S4, 1, [2,3])
        
        out11 = (torch.clip(out11_S1, 0, 1) + torch.clip(out11_S2, 0, 1) + torch.clip(out11_S3, 0, 1) + torch.clip(out11_S4, 0, 1)) / 4.0
        out12 = (torch.clip(out12_S1, 0, 1) + torch.clip(out12_S2, 0, 1) + torch.clip(out12_S3, 0, 1) + torch.clip(out12_S4, 0, 1)) / 4.0
        out21 = (torch.clip(out21_S1, 0, 1) + torch.clip(out21_S2, 0, 1) + torch.clip(out21_S3, 0, 1) + torch.clip(out21_S4, 0, 1)) / 4.0
        out22 = (torch.clip(out22_S1, 0, 1) + torch.clip(out22_S2, 0, 1) + torch.clip(out22_S3, 0, 1) + torch.clip(out22_S4, 0, 1)) / 4.0

        image_l = F.interpolate(image, scale_factor=args.scale, mode='bicubic', align_corners=False)

        out11 = out11 + image_l
        out12 = out12 + image_l
        out21 = out21 + image_l
        out22 = out22 + image_l

        loss11 = F.mse_loss(out11, label)
        loss12 = F.mse_loss(out12, label)
        loss21 = F.mse_loss(out21, label)
        loss22 = F.mse_loss(out22, label)
        loss_D = loss11 + loss12 + loss21 + loss22

        if (diver_w > 0.):
            loss_trip = trip_loss(out11, out12, out21, out22, label, diver_w, args.margin_same, args.margin_diff)
            loss_trip = loss_trip.item()
        else:
            loss_trip = 0.

        loss_divergence = loss_D + loss_trip

        loss_divergence.backward()
        opt_D.step()
        # out11, out12, out21, out22 = model_D(image)

        out11_S1, out12_S1, out21_S1, out22_S1 = model_D(F.pad(image, pad=(0,1,0,1), mode='reflect'))
        out11_S2, out12_S2, out21_S2, out22_S2 = model_D(F.pad(torch.rot90(image, 1, [2,3]), pad=(0,1,0,1), mode='reflect'))
        out11_S2, out12_S2, out21_S2, out22_S2 = torch.rot90(out11_S2, 3, [2,3]), torch.rot90(out12_S2, 3, [2,3]), torch.rot90(out21_S2, 3, [2,3]), torch.rot90(out22_S2, 3, [2,3])
        out11_S3, out12_S3, out21_S3, out22_S3 = model_D(F.pad(torch.rot90(image, 2, [2,3]), pad=(0,1,0,1), mode='reflect'))
        out11_S3, out12_S3, out21_S3, out22_S3 = torch.rot90(out11_S3, 2, [2,3]), torch.rot90(out12_S3, 2, [2,3]), torch.rot90(out21_S3, 2, [2,3]), torch.rot90(out22_S3, 2, [2,3])
        out11_S4, out12_S4, out21_S4, out22_S4 = model_D(F.pad(torch.rot90(image, 3, [2,3]), pad=(0,1,0,1), mode='reflect'))
        out11_S4, out12_S4, out21_S4, out22_S4 = torch.rot90(out11_S4, 1, [2,3]), torch.rot90(out12_S4, 1, [2,3]), torch.rot90(out21_S4, 1, [2,3]), torch.rot90(out22_S4, 1, [2,3])
        
        out11 = (torch.clip(out11_S1, 0, 1) + torch.clip(out11_S2, 0, 1) + torch.clip(out11_S3, 0, 1) + torch.clip(out11_S4, 0, 1)) / 4.0
        out12 = (torch.clip(out12_S1, 0, 1) + torch.clip(out12_S2, 0, 1) + torch.clip(out12_S3, 0, 1) + torch.clip(out12_S4, 0, 1)) / 4.0
        out21 = (torch.clip(out21_S1, 0, 1) + torch.clip(out21_S2, 0, 1) + torch.clip(out21_S3, 0, 1) + torch.clip(out21_S4, 0, 1)) / 4.0
        out22 = (torch.clip(out22_S1, 0, 1) + torch.clip(out22_S2, 0, 1) + torch.clip(out22_S3, 0, 1) + torch.clip(out22_S4, 0, 1)) / 4.0

        return loss_D, loss_trip, loss_divergence, out11, out12, out21, out22


    def train_step_C(out11, out12, out21, out22, label):
        opt_C.zero_grad()
        fusion = model_C(out11, out12, out21, out22)
        image_l = F.interpolate(image, scale_factor=args.scale, mode='bicubic', align_corners=False)

        loss_C = F.mse_loss(fusion+image_l, label)
        loss_C.backward()
        opt_C.step()

        return loss_C

    def valid_step(image, label):
        with torch.no_grad():
            # out11, out12, out21, out22 = model_D(image)
            out11_S1, out12_S1, out21_S1, out22_S1 = model_D(F.pad(image, pad=(0,1,0,1), mode='reflect'))
            out11_S2, out12_S2, out21_S2, out22_S2 = model_D(F.pad(torch.rot90(image, 1, [2,3]), pad=(0,1,0,1), mode='reflect'))
            out11_S2, out12_S2, out21_S2, out22_S2 = torch.rot90(out11_S2, 3, [2,3]), torch.rot90(out12_S2, 3, [2,3]), torch.rot90(out21_S2, 3, [2,3]), torch.rot90(out22_S2, 3, [2,3])
            out11_S3, out12_S3, out21_S3, out22_S3 = model_D(F.pad(torch.rot90(image, 2, [2,3]), pad=(0,1,0,1), mode='reflect'))
            out11_S3, out12_S3, out21_S3, out22_S3 = torch.rot90(out11_S3, 2, [2,3]), torch.rot90(out12_S3, 2, [2,3]), torch.rot90(out21_S3, 2, [2,3]), torch.rot90(out22_S3, 2, [2,3])
            out11_S4, out12_S4, out21_S4, out22_S4 = model_D(F.pad(torch.rot90(image, 3, [2,3]), pad=(0,1,0,1), mode='reflect'))
            out11_S4, out12_S4, out21_S4, out22_S4 = torch.rot90(out11_S4, 1, [2,3]), torch.rot90(out12_S4, 1, [2,3]), torch.rot90(out21_S4, 1, [2,3]), torch.rot90(out22_S4, 1, [2,3])
            
            out11 = (torch.clip(out11_S1, 0, 1) + torch.clip(out11_S2, 0, 1) + torch.clip(out11_S3, 0, 1) + torch.clip(out11_S4, 0, 1)) / 4.0
            out12 = (torch.clip(out12_S1, 0, 1) + torch.clip(out12_S2, 0, 1) + torch.clip(out12_S3, 0, 1) + torch.clip(out12_S4, 0, 1)) / 4.0
            out21 = (torch.clip(out21_S1, 0, 1) + torch.clip(out21_S2, 0, 1) + torch.clip(out21_S3, 0, 1) + torch.clip(out21_S4, 0, 1)) / 4.0
            out22 = (torch.clip(out22_S1, 0, 1) + torch.clip(out22_S2, 0, 1) + torch.clip(out22_S3, 0, 1) + torch.clip(out22_S4, 0, 1)) / 4.0
            sr = model_C(out11, out12, out21, out22)

            image_l = F.interpolate(image, scale_factor=args.scale, mode='bicubic', align_corners=False)
            sr = sr + image_l

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

    def adjust_diver(epoch, diver_init, diver_start_epoch, diver_every, diver_epochs, diver_decay_epoch,
                     diver_decay_rate):
        if (epoch <= diver_start_epoch):
            return 0.
        else:
            diver_w = diver_init * (diver_decay_rate ** (epoch // diver_decay_epoch))
            if (epoch % diver_every < diver_epochs):
                return diver_w
            else:
                return 0.

    # start training
    for step in range(0, int(args.epochs * steps_per_epoch)):
        lr_C = adjust_learning_rate(step, opt_C)
        lr_D = adjust_learning_rate(step, opt_D)
        diver_w = adjust_diver(step // steps_per_epoch, args.diver_w, args.diver_start_epoch, args.diver_every,
                               args.diver_epochs, args.diver_decay_epoch, args.diver_decay_rate)
        t_step = time.time()

        try:
            image, label = next(train_queue)
        except StopIteration:
            train_queue = iter(train_dataloader)
            image, label = next(train_queue)
        
        # image = torch.Tensor(image)
        # label = torch.Tensor(label)
        image, label = image.to(device), label.to(device)

        loss_D, loss_trip, loss_divergence, out11, out12, out21, out22 = train_step_D(image, label)
        loss_C = train_step_C(out11, out12, out21, out22, label)

        t_train = time.time() - t_step
        if step % args.print_freq == 0:
            logger.info(
                "[{}]  Epoch {} Step {} Loss_D={:.5} Loss_trip={:.5} Loss_C={:.5}  lr={:.5}  times={:.2}s".format(
                    args.ex_name,
                    step // steps_per_epoch,
                    step,
                    loss_D.item(),
                    loss_trip,
                    loss_C.item(),
                    lr_D, t_train
                    ))
            print(
                "[{}]  Epoch {} Step {} Loss_D={:.5} Loss_trip={:.5} Loss_C={:.5}  lr={:.5}  times={:.2}s".format(
                    args.ex_name,
                    step // steps_per_epoch,
                    step,
                    loss_D.item(),
                    loss_trip,
                    loss_C.item(),
                    lr_D, t_train
                    ))

            writer.add_scalar('loss_D', loss_D.item(), step)
            writer.add_scalar('loss_Trip', loss_trip, step)
            writer.add_scalar('loss_C', loss_C.item(), step)

        if ((step + 1) % (steps_per_epoch * args.val_freq) == 0):
            model_D.eval()
            model_C.eval()
            
            psnr_v, ssim_v = valid(valid_step, valid_dataloader, len_val)

            model_D.train()
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
            writer.add_scalar('PSNR', psnr_v, step)

            model_D, model_C = model_D.to(device), model_C.to(device)


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