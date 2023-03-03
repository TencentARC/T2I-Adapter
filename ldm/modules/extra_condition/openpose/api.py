import numpy as np
import os
import torch.nn as nn

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import torch

from . import util
from .body import Body

remote_model_path = "https://huggingface.co/TencentARC/T2I-Adapter/blob/main/third-party-models/body_pose_model.pth"


class OpenposeInference(nn.Module):

    def __init__(self):
        super().__init__()
        body_modelpath = os.path.join('models', "body_pose_model.pth")

        if not os.path.exists(body_modelpath):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(remote_model_path, model_dir='models')

        self.body_estimation = Body(body_modelpath)

    def forward(self, x):
        x = x[:, :, ::-1].copy()
        with torch.no_grad():
            candidate, subset = self.body_estimation(x)
            canvas = np.zeros_like(x)
            canvas = util.draw_bodypose(canvas, candidate, subset)
            canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        return canvas
