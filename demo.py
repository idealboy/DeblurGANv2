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


class Predictor:
    def __init__(self, weights_path: str, model_name: str = ''):
        with open('config/config.yaml') as cfg:
            config = yaml.load(cfg)
        model = get_generator(model_name or config['model'])
        model.load_state_dict(torch.load(weights_path)['model'])
        self.model = model.cuda()
        self.model.train(True)
        # GAN inference should be in train mode to use actual stats in norm layers,
        # it's not a bug
        self.normalize_fn = get_normalize()

    def _preprocess(self, x):

        x, _ = self.normalize_fn(x, x)

        h, w, _ = x.shape
        block_size = 32
        min_height = (h // block_size + 1) * block_size
        min_width = (w // block_size + 1) * block_size

        pad_params = {'mode': 'constant',
                      'constant_values': 0,
                      'pad_width': ((0, min_height - h), (0, min_width - w), (0, 0))
                      }
        x = np.pad(x, **pad_params)
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, 0)
        x = torch.from_numpy(x)

        return x, h, w

    def predict(self, img):

        img, h, w = self._preprocess(img)

        with torch.no_grad():
            inputs = [img.cuda()]
            pred = self.model(*inputs)
            pred, =pred
            pred = pred.detach().cpu().float().numpy()
            pred = (np.transpose(pred, (1, 2, 0)) + 1) / 2.0 * 255.0
            cc = pred.astype('uint8')
            cc=cc[:h,:w,:]
            return cc


def main():

    predictor = Predictor(weights_path="fpn_inception.h5")

    out_dir = "submit/"

    os.makedirs(out_dir, exist_ok=True)

    filename = "./test_image/69.png"
    img = cv2.imread(filename)

    pred = predictor.predict(img)

    name = os.path.basename(filename)

    cv2.imwrite(os.path.join(out_dir, name), pred)


if __name__ == '__main__':
    main()
