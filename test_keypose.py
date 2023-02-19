import argparse
import logging
import os
import os.path as osp
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
                           img2tensor, scandir, tensor2img)
from basicsr.utils.options import copy_opt_file, dict2str
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything

from dataset_coco import dataset_coco, dataset_coco_mask_color_sig
from dist_util import get_bare_model, init_dist, master_only
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.modules.encoders.adapter import Adapter
from ldm.util import instantiate_from_config
import mmcv
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import (inference_top_down_pose_model, init_pose_model, process_mmdet_results, vis_pose_result)

skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10],
            [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]

pose_kpt_color = [[51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255], [0, 255, 0],
                  [255, 128, 0], [0, 255, 0], [255, 128, 0], [0, 255, 0], [255, 128, 0], [0, 255, 0], [255, 128, 0],
                  [0, 255, 0], [255, 128, 0], [0, 255, 0], [255, 128, 0]]

pose_link_color = [[0, 255, 0], [0, 255, 0], [255, 128, 0], [255, 128, 0],
                   [51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255], [0, 255, 0], [255, 128, 0],
                   [0, 255, 0], [255, 128, 0], [51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255],
                   [51, 153, 255], [51, 153, 255], [51, 153, 255]]

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

@master_only
def mkdir_and_rename(path):
    """mkdirs. If path exists, rename it with timestamp and create a new one.
    Args:
        path (str): Folder path.
    """
    if osp.exists(path):
        new_name = path + '_archived_' + get_time_str()
        print(f'Path already exists. Rename it to {new_name}', flush=True)
        os.rename(path, new_name)
    os.makedirs(path, exist_ok=True)
    os.makedirs(osp.join(experiments_root, 'models'))
    os.makedirs(osp.join(experiments_root, 'training_states'))
    os.makedirs(osp.join(experiments_root, 'visualization'))

def load_resume_state(opt):
    resume_state_path = None
    if opt.auto_resume:
        state_path = osp.join('experiments', opt.name, 'training_states')
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
                opt.resume_state_path = resume_state_path

    if resume_state_path is None:
        resume_state = None
    else:
        device_id = torch.cuda.current_device()
        resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))

    return resume_state

def imshow_keypoints(img,
                     pose_result,
                     skeleton=None,
                     kpt_score_thr=0.1,
                     pose_kpt_color=None,
                     pose_link_color=None,
                     radius=4,
                     thickness=1):
    """Draw keypoints and links on an image.

    Args:
            img (ndarry): The image to draw poses on.
            pose_result (list[kpts]): The poses to draw. Each element kpts is
                a set of K keypoints as an Kx3 numpy.ndarray, where each
                keypoint is represented as x, y, score.
            kpt_score_thr (float, optional): Minimum score of keypoints
                to be shown. Default: 0.3.
            pose_kpt_color (np.array[Nx3]`): Color of N keypoints. If None,
                the keypoint will not be drawn.
            pose_link_color (np.array[Mx3]): Color of M links. If None, the
                links will not be drawn.
            thickness (int): Thickness of lines.
    """

    img_h, img_w, _ = img.shape
    img = np.zeros(img.shape)

    for idx, kpts in enumerate(pose_result):
        if idx > 1:
            continue
        kpts = kpts['keypoints']
        # print(kpts)
        kpts = np.array(kpts, copy=False)

        # draw each point on image
        if pose_kpt_color is not None:
            assert len(pose_kpt_color) == len(kpts)

            for kid, kpt in enumerate(kpts):
                x_coord, y_coord, kpt_score = int(kpt[0]), int(kpt[1]), kpt[2]

                if kpt_score < kpt_score_thr or pose_kpt_color[kid] is None:
                    # skip the point that should not be drawn
                    continue

                color = tuple(int(c) for c in pose_kpt_color[kid])
                cv2.circle(img, (int(x_coord), int(y_coord)), radius, color, -1)

        # draw links
        if skeleton is not None and pose_link_color is not None:
            assert len(pose_link_color) == len(skeleton)

            for sk_id, sk in enumerate(skeleton):
                pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
                pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))

                if (pos1[0] <= 0 or pos1[0] >= img_w or pos1[1] <= 0 or pos1[1] >= img_h or pos2[0] <= 0
                        or pos2[0] >= img_w or pos2[1] <= 0 or pos2[1] >= img_h or kpts[sk[0], 2] < kpt_score_thr
                        or kpts[sk[1], 2] < kpt_score_thr or pose_link_color[sk_id] is None):
                    # skip the link that should not be drawn
                    continue
                color = tuple(int(c) for c in pose_link_color[sk_id])
                cv2.line(img, pos1, pos2, color, thickness=thickness)

    return img

parser = argparse.ArgumentParser()
parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="An Iron man"
)
parser.add_argument(
        "--neg_prompt",
        type=str,
        default="ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face"
)
parser.add_argument(
        "--path_cond",
        type=str,
        default="examples/keypose/iron.png"
)
parser.add_argument(
        "--type_in",
        type=str,
        default="sketch"
)
parser.add_argument(
    "--bsize",
    type=int,
    default=8,
    help="the prompt to render"
)
parser.add_argument(
    "--epochs",
    type=int,
    default=10000,
    help="the prompt to render"
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda"
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=8,
    help="the prompt to render"
)
parser.add_argument(
    "--use_shuffle",
    type=bool,
    default=True,
    help="the prompt to render"
)
parser.add_argument(
        "--dpm_solver",
        action='store_true',
        help="use dpm_solver sampling",
)
parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
)
parser.add_argument(
        "--auto_resume",
        action='store_true',
        help="use plms sampling",
)
parser.add_argument(
        "--ckpt",
        type=str,
        default="models/sd-v1-4.ckpt",
        help="path to checkpoint of model",
)
parser.add_argument(
        "--ckpt_ad",
        type=str,
        default='models/t2iadapter_keypose_sd14v1.pth'
)
parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/test_keypose.yaml",
        help="path to config which constructs model",
)
parser.add_argument(
        "--print_fq",
        type=int,
        default=100,
        help="path to config which constructs model",
)
parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
)
parser.add_argument(
    "--W",
    type=int,
    default=512,
    help="image width, in pixel space",
)
parser.add_argument(
    "--C",
    type=int,
    default=4,
    help="latent channels",
)
parser.add_argument(
    "--f",
    type=int,
    default=8,
    help="downsampling factor",
)
parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
)
parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
)
parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
)
parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
)
parser.add_argument(
        "--gpus",
        default=[0,1,2,3],
        help="gpu idx",
)
parser.add_argument(
        '--local_rank',
        default=-1,
        type=int,
        help='node rank for distributed training'
)
parser.add_argument(
        '--launcher',
        default='pytorch',
        type=str,
        help='node rank for distributed training'
)

## mmpose part ##
parser.add_argument(
        '--det_config', 
        help='Config file for detection', 
        default='models/faster_rcnn_r50_fpn_coco.py'
)
parser.add_argument(
    '--det_checkpoint',
    help='Checkpoint file for detection',
    default='models/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
)
parser.add_argument(
    '--pose_config',
    help='Config file for pose',
    default='models/hrnet_w48_coco_256x192.py'
)
parser.add_argument(
    '--pose_checkpoint',
    help='Checkpoint file for pose',
    default='models/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
)
parser.add_argument(
    '--det-cat-id', 
    type=int, 
    default=1, 
    help='Category id for bounding box detection model'
)
parser.add_argument(
    '--bbox-thr', 
    type=float, 
    default=0.2, 
    help='Bounding box score threshold'
)

opt = parser.parse_args()

if __name__ == '__main__':
    # seed_everything(42)
    config = OmegaConf.load(f"{opt.config}")
    opt.name = config['name']
    device=opt.device

    # stable diffusion
    model = load_model_from_config(config, f"{opt.ckpt}").to(device)

    # Adaptor
    model_ad = Adapter(cin=int(3*64), channels=[320, 640, 1280, 1280][:4], nums_rb=2, ksize=1, sk=True, use_conv=False).to(device)
    model_ad.load_state_dict(torch.load(opt.ckpt_ad))

    experiments_root = osp.join('experiments', opt.name)

    # resume state
    resume_state = load_resume_state(opt)
    if resume_state is None:
        mkdir_and_rename(experiments_root)

    # copy the yml file to the experiment root
    copy_opt_file(opt.config, experiments_root)

    # WARNING: should not use get_root_logger in the above codes, including the called functions
    # Otherwise the logger will not be properly initialized
    log_file = osp.join(experiments_root, f"train_{opt.name}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(config))

    for v_idx in range(opt.n_samples):
        with torch.no_grad():
            if opt.dpm_solver:
                sampler = DPMSolverSampler(model)
            elif opt.plms:
                sampler = PLMSSampler(model)
            else:
                sampler = DDIMSampler(model)
            c = model.get_learned_conditioning([opt.prompt])

            # costumer input
            if opt.type_in == 'pose':
                pose = cv2.imread(opt.path_cond)
            elif opt.type_in == 'image':
                # im = cv2.imread(opt.path_cond)
                image = cv2.imread(opt.path_cond)
                det_config_mmcv = mmcv.Config.fromfile(opt.det_config)
                det_model = init_detector(det_config_mmcv, opt.det_checkpoint, device=device)
                pose_config_mmcv = mmcv.Config.fromfile(opt.pose_config)
                pose_model = init_pose_model(pose_config_mmcv, opt.pose_checkpoint, device=device)

                mmdet_results = inference_detector(det_model, opt.path_cond)
                # keep the person class bounding boxes.
                person_results = process_mmdet_results(mmdet_results, opt.det_cat_id)

                # optional
                return_heatmap = False
                dataset = pose_model.cfg.data['test']['type']

                # e.g. use ('backbone', ) to return backbone feature
                output_layer_names = None
                pose_results, returned_outputs = inference_top_down_pose_model(
                    pose_model,
                    opt.path_cond,
                    person_results,
                    bbox_thr=opt.bbox_thr,
                    format='xyxy',
                    dataset=dataset,
                    dataset_info=None,
                    return_heatmap=return_heatmap,
                    outputs=output_layer_names)

                # show the results
                pose = imshow_keypoints(
                    image,
                    pose_results,
                    skeleton=skeleton,
                    pose_kpt_color=pose_kpt_color,
                    pose_link_color=pose_link_color,
                    radius=2,
                    thickness=2)

            else:
                raise TypeError('Wrong input condition.')

            pose = cv2.resize(pose,(512,512))
            cv2.imwrite(os.path.join(experiments_root, 'visualization', 'pose_idx%04d.png'%(v_idx)), pose)
            
            pose = img2tensor(pose, bgr2rgb=True, float32=True)/255.
            pose = pose.unsqueeze(0)

            features_adapter = model_ad(pose.to(device))

            shape = [opt.C, opt.H // opt.f, opt.W // opt.f]

            samples_ddim, intermediates = sampler.sample(S=opt.ddim_steps,
                                                conditioning=c,
                                                batch_size=1,
                                                shape=shape,
                                                verbose=False,
                                                unconditional_guidance_scale=opt.scale,
                                                unconditional_conditioning=model.get_learned_conditioning([opt.neg_prompt]),
                                                eta=opt.ddim_eta,
                                                x_T=None,
                                                features_adapter1=features_adapter,
                                                mode = 'pose'
                                                )

            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
            for id_sample, x_sample in enumerate(x_samples_ddim):
                x_sample = 255.*x_sample
                img = x_sample.astype(np.uint8)
                cv2.imwrite(os.path.join(experiments_root, 'visualization', 'sample_idx%04d_s%04d.png'%(v_idx, id_sample)), img[:,:,::-1])