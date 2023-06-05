'''

MegEngine is Licensed under the Apache License, Version 2.0 (the "License")

Copyright (c) Megvii Inc. All rights reserved.
Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

'''

import os, sys
sys.path.append(os.path.abspath('.'))
import numpy as np
import imageio


def stripe_v():
    size = 432
    periods = [4, 8, 12, 16, 32, 48, 72, 96, 144]
    textname = 'dataset/synthetic_0.2.txt'
    text = ''
    for period in periods:
        repeat = size // period
        checker = np.zeros((size, repeat)) + 0.2 # 0.05
        checker[:, 0::2] = 0.8 # 0.95
        checker = np.repeat(checker, period, axis=1)
        checker = (checker*255).astype(np.uint8)
        # breakpoint()
        fname = 'checker_{:03d}.png'.format(period)
        imageio.imwrite('checkerboard_smth/HR/{}'.format(fname), checker)

        text = text + 'checkerboard_smth/LR/{}\t'.format(fname) + 'checkerboard_smth/HR/{}\n'.format(fname)
    with open(textname, 'w') as f:
        f.write(text)


def stripe_h():
    size = 432
    periods = [4, 8, 12, 16, 32, 48] #, 72, 96, 144]
    textname = 'dataset/synthetic_h.txt'
    text = ''
    for period in periods:
        repeat = size // period
        checker = np.zeros((repeat, size)) + 0.05
        checker[0::2] = 0.95
        checker = np.repeat(checker, period, axis=0)
        checker = (checker*255).astype(np.uint8)
        # breakpoint()
        fname = 'checker_{:03d}.png'.format(period)
        imageio.imwrite('checkerboard_h/HR/{}'.format(fname), checker)

        text = text + 'checkerboard_h/LR/{}\t'.format(fname) + 'checkerboard_h/HR/{}\n'.format(fname)
    with open(textname, 'w') as f:
        f.write(text)


def checkerboard():
    size = 432
    periods = [4, 8, 12, 16, 32, 48]
    textname = 'dataset/synthetic_chess.txt'

    for period in periods:
        repeat = size // period
        a = np.resize([0.05, 0.95], repeat)
        checker = np.abs(a-np.array([a]).T)
        checker = np.repeat(checker, period, axis=0)
        checker = np.repeat(checker, period, axis=1)
        checker = (checker*255).astype(np.uint8)
        fname = 'checker_{:03d}.png'.format(period)
        imageio.imwrite('checkerboard_chess/HR/{}'.format(fname), checker)
        text = text + 'checkerboard_chess/LR/{}\t'.format(fname) + 'checkerboard_chess/HR/{}\n'.format(fname)
    
    with open(textname, 'w') as f:
        f.write(text)

def stripe_blueyellow_v():
    size = 432
    periods = [4, 8, 12, 16, 32, 48]
    textname = 'dataset/synthetic_by.txt'
    text = ''
    for period in periods:
        repeat = size // period
        checker = np.zeros((size, repeat)) + 0.1
        checker[:, 0::2, 0] = 0.9
        checker[:, 0::2, 1] = 0.9
        checker[:, 1::2, 2] = 0.9

        checker = np.repeat(checker, period, axis=1)
        checker = (checker*255).astype(np.uint8)
        # breakpoint()
        fname = 'checker_{:03d}.png'.format(period)
        imageio.imwrite('checkerboard_by/HR/{}'.format(fname), checker)

        text = text + 'checkerboard_by/LR/{}\t'.format(fname) + 'checkerboard_by/HR/{}\n'.format(fname)
    with open(textname, 'w') as f:
        f.write(text)


if __name__ == "__main__":
    # main()
    # stripe_h()
    checkerboard()
