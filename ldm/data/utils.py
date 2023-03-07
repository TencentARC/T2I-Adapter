# -*- coding: utf-8 -*-

import cv2
import numpy as np
from torchvision.transforms import transforms
from torchvision.transforms.functional import to_tensor
from transformers import CLIPProcessor

from basicsr.utils import img2tensor


class AddCannyFreezeThreshold(object):

    def __init__(self, low_threshold=100, high_threshold=200):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def __call__(self, sample):
        # sample['jpg'] is PIL image
        x = sample['jpg']
        img = cv2.cvtColor(np.array(x), cv2.COLOR_RGB2BGR)
        canny = cv2.Canny(img, self.low_threshold, self.high_threshold)[..., None]
        sample['canny'] = img2tensor(canny, bgr2rgb=True, float32=True) / 255.
        sample['jpg'] = to_tensor(x)
        return sample


class AddCannyRandomThreshold(object):

    def __init__(self, low_threshold=100, high_threshold=200, shift_range=50):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.threshold_prng = np.random.RandomState()
        self.shift_range = shift_range

    def __call__(self, sample):
        # sample['jpg'] is PIL image
        x = sample['jpg']
        img = cv2.cvtColor(np.array(x), cv2.COLOR_RGB2BGR)
        low_threshold = self.low_threshold + self.threshold_prng.randint(-self.shift_range, self.shift_range)
        high_threshold = self.high_threshold + self.threshold_prng.randint(-self.shift_range, self.shift_range)
        canny = cv2.Canny(img, low_threshold, high_threshold)[..., None]
        sample['canny'] = img2tensor(canny, bgr2rgb=True, float32=True) / 255.
        sample['jpg'] = to_tensor(x)
        return sample


class AddStyle(object):

    def __init__(self, version):
        self.processor = CLIPProcessor.from_pretrained(version)
        self.pil_to_tensor = transforms.ToTensor()

    def __call__(self, sample):
        # sample['jpg'] is PIL image
        x = sample['jpg']
        style = self.processor(images=x, return_tensors="pt")['pixel_values'][0]
        sample['style'] = style
        sample['jpg'] = to_tensor(x)
        return sample
