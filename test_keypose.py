import argparse
import os

import cv2
import numpy as np
import torch
from torch import autocast
from basicsr.utils import img2tensor, tensor2img
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything

from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.encoders.adapter import Adapter
from ldm.util import load_model_from_config, resize_numpy_image, fix_cond_shapes
from ldm.modules.structure_condition.utils import imshow_keypoints

torch.set_grad_enabled(False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/test-keypose"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="An Iron man"
    )
    parser.add_argument(
        "--neg_prompt",
        type=str,
        default="lowres, worst quality, low quality, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face"
    )
    parser.add_argument(
        "--path_cond",
        type=str,
        default="examples/keypose/iron.png"
    )
    parser.add_argument(
        "--type_in",
        type=str,
        default="pose"
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="plms"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/sd-v1-4.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--ckpt_vae",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--ckpt_ad",
        type=str,
        default="models/t2iadapter_keypose_sd14v1.pth"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/sd-v1-inference.yaml",
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
        "--steps",
        type=int,
        default=50,
        help="number of sampling steps",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        '--cond_tau',
        type=float,
        default=1.0,
        help='timestamp parameter that determines until which step the adapter is applied, similar as Prompt-to-Prompt tau'
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    # --- mmpose part --- #
    parser.add_argument(
        '--det_config',
        help='Config file for detection',
        default='configs/mm/faster_rcnn_r50_fpn_coco.py'
    )
    parser.add_argument(
        '--det_checkpoint',
        help='Checkpoint file for detection',
        default='models/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    )
    parser.add_argument(
        '--pose_config',
        help='Config file for pose',
        default='configs/mm/hrnet_w48_coco_256x192.py'
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
    parser.add_argument(
        "--max_resolution",
        type=float,
        default=512 * 512,
        help="image height * width",
    )

    opt = parser.parse_args()
    return opt


def main(opt):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # SD
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, opt.ckpt, opt.ckpt_vae)
    model = model.to(device)

    # Adaptor
    model_ad = Adapter(cin=int(3 * 64), channels=[320, 640, 1280, 1280][:4], nums_rb=2, ksize=1, sk=True,
                       use_conv=False).to(device)
    model_ad.load_state_dict(torch.load(opt.ckpt_ad))

    # det model
    if opt.type_in == 'image':
        import mmcv
        from mmdet.apis import inference_detector, init_detector
        from mmpose.apis import inference_top_down_pose_model, init_pose_model, process_mmdet_results
        det_config_mmcv = mmcv.Config.fromfile(opt.det_config)
        det_model = init_detector(det_config_mmcv, opt.det_checkpoint, device=device)
        pose_config_mmcv = mmcv.Config.fromfile(opt.pose_config)
        pose_model = init_pose_model(pose_config_mmcv, opt.pose_checkpoint, device=device)

    # sampler
    if opt.sampler == 'plms':
        sampler = PLMSSampler(model)
    elif opt.sampler == 'ddim':
        sampler = DDIMSampler(model)
    else:
        raise NotImplementedError

    os.makedirs(opt.outdir, exist_ok=True)

    seed_everything(opt.seed)

    with torch.no_grad(), \
            model.ema_scope():
        for v_idx in range(opt.n_samples):
            if opt.type_in == 'pose':
                pose = cv2.imread(opt.path_cond)
                pose = resize_numpy_image(pose, max_resolution=opt.max_resolution)
            elif opt.type_in == 'image':
                image = cv2.imread(opt.path_cond)
                image = resize_numpy_image(image, max_resolution=opt.max_resolution)
                mmdet_results = inference_detector(det_model, image)
                # keep the person class bounding boxes.
                person_results = process_mmdet_results(mmdet_results, opt.det_cat_id)

                # optional
                return_heatmap = False
                dataset = pose_model.cfg.data['test']['type']

                # e.g. use ('backbone', ) to return backbone feature
                output_layer_names = None
                pose_results, returned_outputs = inference_top_down_pose_model(
                    pose_model,
                    image,
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
                    radius=2,
                    thickness=2)
            else:
                raise TypeError('Wrong input condition.')

            opt.H, opt.W = pose.shape[:2]

            with autocast('cuda'):
                c = model.get_learned_conditioning([opt.prompt])
                if opt.scale != 1.0:
                    uc = model.get_learned_conditioning([opt.neg_prompt])
                else:
                    uc = None
                c, uc = fix_cond_shapes(model, c, uc)

                base_count = len(os.listdir(opt.outdir)) // 2

                cv2.imwrite(os.path.join(opt.outdir, f'{base_count:05}_pose.png'), pose)

                pose = img2tensor(pose, bgr2rgb=True, float32=True) / 255.
                pose = pose.unsqueeze(0)

                features_adapter = model_ad(pose.to(device))

                shape = [opt.C, opt.H // opt.f, opt.W // opt.f]

                samples_ddim, _ = sampler.sample(S=opt.steps,
                                                 conditioning=c,
                                                 batch_size=1,
                                                 shape=shape,
                                                 verbose=False,
                                                 unconditional_guidance_scale=opt.scale,
                                                 unconditional_conditioning=uc,
                                                 x_T=None,
                                                 features_adapter=features_adapter,
                                                 cond_tau=opt.cond_tau
                                                 )

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                x_samples_ddim = x_samples_ddim.permute(0, 2, 3, 1)[0].cpu().numpy()
                x_sample = 255. * x_samples_ddim
                x_sample = Image.fromarray(x_sample.astype(np.uint8))
                x_sample.save(os.path.join(opt.outdir, f'{base_count:05}_result.png'))


if __name__ == '__main__':
    opt = parse_args()
    main(opt)
