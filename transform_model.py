import os
from glob import glob
from typing import Optional

import cv2
import numpy as np
import torch
import yaml
from fire import Fire
from tqdm import tqdm

from aug import get_normalize
from models.networks import get_generator
from models.fpn_inception_script import FPNInceptionScript
import functools
import torch.nn as nn
from collections import OrderedDict

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

class Predictor:
    def __init__(self, weights_path):
        with open('config/config.yaml') as cfg:
            config = yaml.load(cfg)
        model = FPNInceptionScript(norm_layer=get_norm_layer(norm_type=config['model']['norm_layer']), output_ch = 3, num_filters = 128, num_filters_fpn = 256)
        state_dict = torch.load(weights_path)['model']

        new_state_dict = OrderedDict()



        for k,v in state_dict.items():
            name = k
            name = name.replace("module.", "")
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
        model = model.cuda()
        example = torch.rand((1, 3, 224, 224)).cuda()
        traced_script_module = torch.jit.script(model)
        traced_script_module.save("deblur_fpn_inception.pt")

def main():

    predictor = Predictor(weights_path="fpn_inception.h5")

if __name__ == '__main__':
    main()
