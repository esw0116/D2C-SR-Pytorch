
import os, glob, sys
sys.path.append(os.path.abspath('.'))

import numpy as np
from tqdm import tqdm
import logging
import imageio

from model.simple5_model import *
import torch
import torch.utils.data as data

from config.config_test import args
from dataset.dataset import *
from metrics.PSNR import *
from utils import eval_metrics


logger = logging.getLogger()

def FourSimplexInterp(weight, img_in, h, w, q, rot, upscale=4):
    SAMPLING_INTERVAL = 4   # N bit uniform sampling
    L = 2**(8-SAMPLING_INTERVAL) + 1

    # Extract MSBs
    img_a1 = img_in[:, 0:0+h, 0:0+w] // q
    img_b1 = img_in[:, 0:0+h, 1:1+w] // q
    img_c1 = img_in[:, 1:1+h, 0:0+w] // q
    img_d1 = img_in[:, 1:1+h, 1:1+w] // q
    
    img_a2 = img_a1 + 1
    img_b2 = img_b1 + 1
    img_c2 = img_c1 + 1
    img_d2 = img_d1 + 1

    # Extract LSBs
    fa_ = img_in[:, 0:0+h, 0:0+w] % q
    fb_ = img_in[:, 0:0+h, 1:1+w] % q
    fc_ = img_in[:, 1:1+h, 0:0+w] % q
    fd_ = img_in[:, 1:1+h, 1:1+w] % q


    # Vertices (O in Eq3 and Table3 in the paper)
    p0000 = weight[ img_a1.flatten().astype(np.int_)*L*L*L + img_b1.flatten().astype(np.int_)*L*L + img_c1.flatten().astype(np.int_)*L + img_d1.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p0001 = weight[ img_a1.flatten().astype(np.int_)*L*L*L + img_b1.flatten().astype(np.int_)*L*L + img_c1.flatten().astype(np.int_)*L + img_d2.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p0010 = weight[ img_a1.flatten().astype(np.int_)*L*L*L + img_b1.flatten().astype(np.int_)*L*L + img_c2.flatten().astype(np.int_)*L + img_d1.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p0011 = weight[ img_a1.flatten().astype(np.int_)*L*L*L + img_b1.flatten().astype(np.int_)*L*L + img_c2.flatten().astype(np.int_)*L + img_d2.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p0100 = weight[ img_a1.flatten().astype(np.int_)*L*L*L + img_b2.flatten().astype(np.int_)*L*L + img_c1.flatten().astype(np.int_)*L + img_d1.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p0101 = weight[ img_a1.flatten().astype(np.int_)*L*L*L + img_b2.flatten().astype(np.int_)*L*L + img_c1.flatten().astype(np.int_)*L + img_d2.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p0110 = weight[ img_a1.flatten().astype(np.int_)*L*L*L + img_b2.flatten().astype(np.int_)*L*L + img_c2.flatten().astype(np.int_)*L + img_d1.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p0111 = weight[ img_a1.flatten().astype(np.int_)*L*L*L + img_b2.flatten().astype(np.int_)*L*L + img_c2.flatten().astype(np.int_)*L + img_d2.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    
    p1000 = weight[ img_a2.flatten().astype(np.int_)*L*L*L + img_b1.flatten().astype(np.int_)*L*L + img_c1.flatten().astype(np.int_)*L + img_d1.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p1001 = weight[ img_a2.flatten().astype(np.int_)*L*L*L + img_b1.flatten().astype(np.int_)*L*L + img_c1.flatten().astype(np.int_)*L + img_d2.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p1010 = weight[ img_a2.flatten().astype(np.int_)*L*L*L + img_b1.flatten().astype(np.int_)*L*L + img_c2.flatten().astype(np.int_)*L + img_d1.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p1011 = weight[ img_a2.flatten().astype(np.int_)*L*L*L + img_b1.flatten().astype(np.int_)*L*L + img_c2.flatten().astype(np.int_)*L + img_d2.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p1100 = weight[ img_a2.flatten().astype(np.int_)*L*L*L + img_b2.flatten().astype(np.int_)*L*L + img_c1.flatten().astype(np.int_)*L + img_d1.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p1101 = weight[ img_a2.flatten().astype(np.int_)*L*L*L + img_b2.flatten().astype(np.int_)*L*L + img_c1.flatten().astype(np.int_)*L + img_d2.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p1110 = weight[ img_a2.flatten().astype(np.int_)*L*L*L + img_b2.flatten().astype(np.int_)*L*L + img_c2.flatten().astype(np.int_)*L + img_d1.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    p1111 = weight[ img_a2.flatten().astype(np.int_)*L*L*L + img_b2.flatten().astype(np.int_)*L*L + img_c2.flatten().astype(np.int_)*L + img_d2.flatten().astype(np.int_) ].reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    
    # Output image holder
    out = np.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))

    # Naive pixelwise output value interpolation (Table3 in the paper)
    # It would be faster implemented with a parallel operation
    for c in range(img_a1.shape[0]):
        for y in range(img_a1.shape[1]):
            for x in range(img_a1.shape[2]):
                fa = fa_[c,y,x]
                fb = fb_[c,y,x]
                fc = fc_[c,y,x]
                fd = fd_[c,y,x]

                if fa > fb:
                    if fb > fc:
                        if fc > fd:
                            out[c,y,x] = (q-fa) * p0000[c,y,x] + (fa-fb) * p1000[c,y,x] + (fb-fc) * p1100[c,y,x] + (fc-fd) * p1110[c,y,x] + (fd) * p1111[c,y,x]
                        elif fb > fd:
                            out[c,y,x] = (q-fa) * p0000[c,y,x] + (fa-fb) * p1000[c,y,x] + (fb-fd) * p1100[c,y,x] + (fd-fc) * p1101[c,y,x] + (fc) * p1111[c,y,x]
                        elif fa > fd:
                            out[c,y,x] = (q-fa) * p0000[c,y,x] + (fa-fd) * p1000[c,y,x] + (fd-fb) * p1001[c,y,x] + (fb-fc) * p1101[c,y,x] + (fc) * p1111[c,y,x]
                        else:
                            out[c,y,x] = (q-fd) * p0000[c,y,x] + (fd-fa) * p0001[c,y,x] + (fa-fb) * p1001[c,y,x] + (fb-fc) * p1101[c,y,x] + (fc) * p1111[c,y,x]
                    elif fa > fc:
                        if fb > fd:
                            out[c,y,x] = (q-fa) * p0000[c,y,x] + (fa-fc) * p1000[c,y,x] + (fc-fb) * p1010[c,y,x] + (fb-fd) * p1110[c,y,x] + (fd) * p1111[c,y,x]
                        elif fc > fd:
                            out[c,y,x] = (q-fa) * p0000[c,y,x] + (fa-fc) * p1000[c,y,x] + (fc-fd) * p1010[c,y,x] + (fd-fb) * p1011[c,y,x] + (fb) * p1111[c,y,x]
                        elif fa > fd:
                            out[c,y,x] = (q-fa) * p0000[c,y,x] + (fa-fd) * p1000[c,y,x] + (fd-fc) * p1001[c,y,x] + (fc-fb) * p1011[c,y,x] + (fb) * p1111[c,y,x]
                        else:
                            out[c,y,x] = (q-fd) * p0000[c,y,x] + (fd-fa) * p0001[c,y,x] + (fa-fc) * p1001[c,y,x] + (fc-fb) * p1011[c,y,x] + (fb) * p1111[c,y,x]
                    else:
                        if fb > fd:
                            out[c,y,x] = (q-fc) * p0000[c,y,x] + (fc-fa) * p0010[c,y,x] + (fa-fb) * p1010[c,y,x] + (fb-fd) * p1110[c,y,x] + (fd) * p1111[c,y,x]
                        elif fc > fd:
                            out[c,y,x] = (q-fc) * p0000[c,y,x] + (fc-fa) * p0010[c,y,x] + (fa-fd) * p1010[c,y,x] + (fd-fb) * p1011[c,y,x] + (fb) * p1111[c,y,x]
                        elif fa > fd:
                            out[c,y,x] = (q-fc) * p0000[c,y,x] + (fc-fd) * p0010[c,y,x] + (fd-fa) * p0011[c,y,x] + (fa-fb) * p1011[c,y,x] + (fb) * p1111[c,y,x]
                        else:
                            out[c,y,x] = (q-fd) * p0000[c,y,x] + (fd-fc) * p0001[c,y,x] + (fc-fa) * p0011[c,y,x] + (fa-fb) * p1011[c,y,x] + (fb) * p1111[c,y,x]

                else:
                    if fa > fc:
                        if fc > fd:
                            out[c,y,x] = (q-fb) * p0000[c,y,x] + (fb-fa) * p0100[c,y,x] + (fa-fc) * p1100[c,y,x] + (fc-fd) * p1110[c,y,x] + (fd) * p1111[c,y,x]
                        elif fa > fd:
                            out[c,y,x] = (q-fb) * p0000[c,y,x] + (fb-fa) * p0100[c,y,x] + (fa-fd) * p1100[c,y,x] + (fd-fc) * p1101[c,y,x] + (fc) * p1111[c,y,x]
                        elif fb > fd:
                            out[c,y,x] = (q-fb) * p0000[c,y,x] + (fb-fd) * p0100[c,y,x] + (fd-fa) * p0101[c,y,x] + (fa-fc) * p1101[c,y,x] + (fc) * p1111[c,y,x]
                        else:
                            out[c,y,x] = (q-fd) * p0000[c,y,x] + (fd-fb) * p0001[c,y,x] + (fb-fa) * p0101[c,y,x] + (fa-fc) * p1101[c,y,x] + (fc) * p1111[c,y,x]
                    elif fb > fc:
                        if fa > fd:
                            out[c,y,x] = (q-fb) * p0000[c,y,x] + (fb-fc) * p0100[c,y,x] + (fc-fa) * p0110[c,y,x] + (fa-fd) * p1110[c,y,x] + (fd) * p1111[c,y,x]
                        elif fc > fd:
                            out[c,y,x] = (q-fb) * p0000[c,y,x] + (fb-fc) * p0100[c,y,x] + (fc-fd) * p0110[c,y,x] + (fd-fa) * p0111[c,y,x] + (fa) * p1111[c,y,x]
                        elif fb > fd:
                            out[c,y,x] = (q-fb) * p0000[c,y,x] + (fb-fd) * p0100[c,y,x] + (fd-fc) * p0101[c,y,x] + (fc-fa) * p0111[c,y,x] + (fa) * p1111[c,y,x]
                        else:
                            out[c,y,x] = (q-fd) * p0000[c,y,x] + (fd-fb) * p0001[c,y,x] + (fb-fc) * p0101[c,y,x] + (fc-fa) * p0111[c,y,x] + (fa) * p1111[c,y,x]
                    else:
                        if fa > fd:
                            out[c,y,x] = (q-fc) * p0000[c,y,x] + (fc-fb) * p0010[c,y,x] + (fb-fa) * p0110[c,y,x] + (fa-fd) * p1110[c,y,x] + (fd) * p1111[c,y,x]
                        elif fb > fd:
                            out[c,y,x] = (q-fc) * p0000[c,y,x] + (fc-fb) * p0010[c,y,x] + (fb-fd) * p0110[c,y,x] + (fd-fa) * p0111[c,y,x] + (fa) * p1111[c,y,x]
                        elif fc > fd:
                            out[c,y,x] = (q-fc) * p0000[c,y,x] + (fc-fd) * p0010[c,y,x] + (fd-fb) * p0011[c,y,x] + (fb-fa) * p0111[c,y,x] + (fa) * p1111[c,y,x]
                        else:
                            out[c,y,x] = (q-fd) * p0000[c,y,x] + (fd-fc) * p0001[c,y,x] + (fc-fb) * p0011[c,y,x] + (fb-fa) * p0111[c,y,x] + (fa) * p1111[c,y,x]

    out = np.transpose(out, (0, 1,3, 2,4)).reshape((img_a1.shape[0], img_a1.shape[1]*upscale, img_a1.shape[2]*upscale))
    out = np.rot90(out, rot, [1,2])
    out = out / q
    return out

def lut_interp(img_lr):
    UPSCALE = 4       # upscaling factor
    SAMPLING_INTERVAL = 4   # N bit uniform sampling
    q = 2**SAMPLING_INTERVAL

    img_lr = img_lr.squeeze(0).cpu().numpy()
    c, h, w = img_lr.shape

    img_outs = []
    for idx in range(4):
        LUT_PATH = "experiments/pytorch_simple5_triploss_x4/LUTs/Model_D_{}_x{}_{}bit.npy".format(idx, UPSCALE, SAMPLING_INTERVAL)    # Trained SR net params
        LUT = np.load(LUT_PATH).astype(np.float32).reshape(-1, UPSCALE*UPSCALE)  # N(=(2^SAMPLING_INTERVAL + 1)^4D), 16(=r*r)
        
        img_in = np.pad(img_lr, ((0,0), (0,1), (0,1)), mode='reflect')
        img_out = FourSimplexInterp(LUT, img_in, h, w, q, 0, upscale=UPSCALE)

        img_out = img_out[np.newaxis, :, :, :]
        img_out = torch.Tensor(np.clip(img_out, 0, 1)).to('cuda')
        # img_out = np.round(np.clip(img_out, 0, 1) * 255).astype(np.uint8)
        img_outs.append(img_out)
    
    return img_outs


def main(args):
    SAMPLING_INTERVAL = 4   # N bit uniform sampling
    device = 'cuda'

    # create dataset
    valid_dataloader, len_val = create_dataset(args)

    if not os.path.exists(os.path.join('output_D2C_x{}_{}bit'.format(args.scale, SAMPLING_INTERVAL), 'output')):
        os.makedirs(os.path.join('output_D2C_x{}_{}bit'.format(args.scale, SAMPLING_INTERVAL), 'output'))
    if not os.path.exists(os.path.join('output_D2C_x{}_{}bit'.format(args.scale, SAMPLING_INTERVAL), 'mask')):
        os.makedirs(os.path.join('output_D2C_x{}_{}bit'.format(args.scale, SAMPLING_INTERVAL), 'mask'))

    # args = {}
    # args['scale'] = 4
    # model_D = D_Net(args)
    # model_D = D_Net(scale=UPSCALE)
    model_C = C_Net(details=True)
    lm = torch.load(args.checkpoint)
    print("model path: %s"%(args.checkpoint))
    # model_D.load_state_dict(lm["state_dict_D"])
    model_C.load_state_dict(lm["state_dict_C"])
    model_C = model_C.to(device)
    
    def valid_step(image, label, filename):
        scale = 4
        img_d = lut_interp(image)
        with torch.no_grad():
            sr, mask = model_C(*img_d)
        ## Saving SR
        sr_np = sr.cpu().numpy()
        sr_np = sr_np[0].transpose(1,2,0)
        sr_np = sr_np[:,:, [2,1,0]]
        imageio.imwrite('./output_D2C_x{}_{}bit/output/{}_LUT_interp.png'.format(scale, SAMPLING_INTERVAL, filename[0].split('/')[-1][:-4]), sr_np)
        ## Saving mask
        mask_1 = mask[:,0:1,...].cpu().numpy()[0].transpose((1,2,0))
        mask_2 = mask[:,1:2,...].cpu().numpy()[0].transpose((1,2,0))
        mask_3 = mask[:,2:3,...].cpu().numpy()[0].transpose((1,2,0))
        mask_4 = mask[:,3:,...].cpu().numpy()[0].transpose((1,2,0))

        imageio.imwrite('./output_D2C_x{}_{}bit/mask/{}_LUT_mask_1.png'.format(scale, SAMPLING_INTERVAL, filename[0].split('/')[-1][:-4]), mask_1)
        imageio.imwrite('./output_D2C_x{}_{}bit/mask/{}_LUT_mask_2.png'.format(scale, SAMPLING_INTERVAL, filename[0].split('/')[-1][:-4]), mask_2)
        imageio.imwrite('./output_D2C_x{}_{}bit/mask/{}_LUT_mask_3.png'.format(scale, SAMPLING_INTERVAL, filename[0].split('/')[-1][:-4]), mask_3)
        imageio.imwrite('./output_D2C_x{}_{}bit/mask/{}_LUT_mask_4.png'.format(scale, SAMPLING_INTERVAL, filename[0].split('/')[-1][:-4]), mask_4)

        sr = sr.cpu(); label = label.cpu()
        _, ssim_it = eval_metrics(sr, label)
        psnr_it = cal_psnr(sr, label, scale)
        return psnr_it, ssim_it

    # model_D.eval()
    model_C.eval()
    psnr_v, ssim_v = valid(valid_step, valid_dataloader, len_val)
    logger.info("PSNR [\033[1;31m{:.2f}\033[0m]  SSIM [\033[1;31m{:.3f}\033[0m] ".format(psnr_v, ssim_v))


def valid(func, data_queue, len_val):
    device = 'cuda'

    psnr_v = 0.
    ssim_v = 0.
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

    return valid_dataloader, len_val

if __name__ == "__main__":
    main(args)
"""
# USER PARAMS
UPSCALE = 4             # upscaling factor
SAMPLING_INTERVAL = 4   # N bit uniform sampling
idx = 0
LUT_PATH = "Model_D_{}_x{}_{}bit_int8.npy".format(idx, UPSCALE, SAMPLING_INTERVAL)    # Trained SR net params
TEST_DIR = './test/'      # Test images


# Load LUT
LUT = np.load(LUT_PATH).astype(np.float32).reshape(-1, UPSCALE*UPSCALE)  # N(=(2^SAMPLING_INTERVAL + 1)^4D), 16(=r*r)



# Test LR images
files_lr = glob.glob(TEST_DIR + '/LR_x{}/*.png'.format(UPSCALE))
files_lr.sort()

# Test GT images
files_gt = glob.glob(TEST_DIR + '/HR/*.png')
files_gt.sort()


psnrs = []

if not isdir('./output_S_x{}_{}bit'.format(UPSCALE, SAMPLING_INTERVAL)):
    mkdir('./output_S_x{}_{}bit'.format(UPSCALE, SAMPLING_INTERVAL))

for ti, fn in enumerate(tqdm(files_gt)):
    # Load LR image
    img_lr = np.array(Image.open(files_lr[ti])).astype(np.float32)
    h, w, c = img_lr.shape

    # Load GT image
    img_gt = np.array(Image.open(files_gt[ti]))
    
    # Sampling interval for input

    
    # Rotational ensemble
    img_in = np.pad(img_lr, ((0,1), (0,1), (0,0)), mode='reflect').transpose((2,0,1))
    out_r0 = FourSimplexInterp(LUT, img_in, h, w, q, 0, upscale=UPSCALE)

    img_in = np.pad(np.rot90(img_lr, 1), ((0,1), (0,1), (0,0)), mode='reflect').transpose((2,0,1))
    out_r1 = FourSimplexInterp(LUT, img_in, w, h, q, 3, upscale=UPSCALE)

    img_in = np.pad(np.rot90(img_lr, 2), ((0,1), (0,1), (0,0)), mode='reflect').transpose((2,0,1))
    out_r2 = FourSimplexInterp(LUT, img_in, h, w, q, 2, upscale=UPSCALE)

    img_in = np.pad(np.rot90(img_lr, 3), ((0,1), (0,1), (0,0)), mode='reflect').transpose((2,0,1))
    out_r3 = FourSimplexInterp(LUT, img_in, w, h, q, 1, upscale=UPSCALE)

    img_out = (out_r0/1.0 + out_r1/1.0 + out_r2/1.0 + out_r3/1.0) / 255.0
    img_out = img_out.transpose((1,2,0))
    img_out = np.round(np.clip(img_out, 0, 1) * 255).astype(np.uint8)

    # Matching image sizes 
    if img_gt.shape[0] < img_out.shape[0]:
        img_out = img_out[:img_gt.shape[0]]
    if img_gt.shape[1] < img_out.shape[1]:
        img_out = img_out[:, :img_gt.shape[1]]

    if img_gt.shape[0] > img_out.shape[0]:
        img_out = np.pad(img_out, ((0,img_gt.shape[0]-img_out.shape[0]),(0,0),(0,0)))
    if img_gt.shape[1] > img_out.shape[1]:
        img_out = np.pad(img_out, ((0,0),(0,img_gt.shape[1]-img_out.shape[1]),(0,0)))

    # Save to file
    Image.fromarray(img_out).save('./output_S_x{}_{}bit/{}_LUT_interp_{}bit.png'.format(UPSCALE, SAMPLING_INTERVAL, fn.split('/')[-1][:-4], SAMPLING_INTERVAL))

    CROP_S = 4
    psnr = PSNR(_rgb2ycbcr(img_gt)[:,:,0], _rgb2ycbcr(img_out)[:,:,0], CROP_S)
    psnrs.append(psnr)

print('AVG PSNR: {}'.format(np.mean(np.asarray(psnrs))))
"""