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

condition_types = ['sketch', 'seg', 'pose', 'depth']

torch.set_grad_enabled(False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/test-composable"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="A car with flying wings"
    )
    parser.add_argument(
        "--neg_prompt",
        type=str,
        default="ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face"
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
        "--config",
        type=str,
        default="configs/stable-diffusion/sd-v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--max_resolution",
        type=float,
        default=512 * 512,
        help="image height * width",
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
        default=0.4,
        help='timestamp parameter that determines until which step the adapter is applied, similar as Prompt-to-Prompt tau'
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    # multiple structure conditions
    parser.add_argument(
        "--sketch_cond_path",
        type=str,
        default=None
    )
    parser.add_argument(
        "--sketch_cond_weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--sketch_ckpt",
        type=str,
        default="models/t2iadapter_sketch_sd14v1.pth"
    )
    parser.add_argument(
        "--sketch_type_in",
        type=str,
        default="sketch",
    )
    parser.add_argument(
        "--seg_cond_path",
        type=str,
        default=None
    )
    parser.add_argument(
        "--seg_cond_weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--seg_ckpt",
        type=str,
        default="models/t2iadapter_seg_sd14v1.pth"
    )
    parser.add_argument(
        "--seg_type_in",
        type=str,
        default="seg"
    )
    parser.add_argument(
        "--pose_cond_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--pose_cond_weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--pose_ckpt",
        type=str,
        default="models/t2iadapter_keypose_sd14v1.pth"
    )
    parser.add_argument(
        "--pose_type_in",
        type=str,
        default="pose"
    )
    parser.add_argument(
        "--depth_cond_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--depth_cond_weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--depth_ckpt",
        type=str,
        default="models/t2iadapter_depth_sd14v1.pth",
    )
    parser.add_argument(
        "--depth_type_in",
        type=str,
        default="depth"
    )
    opt = parser.parse_args()
    return opt


def get_cond_sketch(opt, cond_path, cond_type_in, cond_preprocess):
    edge = cv2.imread(cond_path)
    edge = resize_numpy_image(edge, max_resolution=opt.max_resolution)
    opt.H, opt.W = edge.shape[:2]
    if cond_type_in == 'sketch':
        edge = img2tensor(edge)[0].unsqueeze(0).unsqueeze(0) / 255.
        edge = edge.to(opt.device)
    elif cond_type_in == 'image':
        if 'sketch' not in cond_preprocess:
            from ldm.modules.structure_condition.model_edge import pidinet
            cond_preprocess['sketch'] = pidinet()
            ckp = torch.load('models/table5_pidinet.pth', map_location='cpu')['state_dict']
            cond_preprocess['sketch'].load_state_dict({k.replace('module.', ''): v for k, v in ckp.items()},
                                                      strict=True)
            cond_preprocess['sketch'].to(opt.device)
        edge = img2tensor(edge).unsqueeze(0) / 255.
        edge = cond_preprocess['sketch'](edge.to(opt.device))[-1]
    else:
        raise NotImplementedError

    # edge = 1-edge # for white background
    edge = edge > 0.5
    edge = edge.float()

    return edge


def get_cond_seg(opt, cond_path, cond_type_in, cond_preprocess):
    seg = cv2.imread(cond_path)
    seg = resize_numpy_image(seg, max_resolution=opt.max_resolution)
    opt.H, opt.W = seg.shape[:2]
    if cond_type_in == 'seg':
        seg = img2tensor(seg).unsqueeze(0) / 255.
        seg = seg.to(opt.device)
    else:
        raise NotImplementedError

    return seg


def get_cond_pose(opt, cond_path, cond_type_in, cond_preprocess):
    pose = cv2.imread(cond_path)
    pose = resize_numpy_image(pose, max_resolution=opt.max_resolution)
    opt.H, opt.W = pose.shape[:2]
    if cond_type_in == 'pose':
        pose = img2tensor(pose).unsqueeze(0) / 255.
        pose = pose.to(opt.device)
    else:
        raise NotImplementedError

    return pose

def get_cond_depth(opt, cond_path, cond_type_in, cond_preprocess):
    depth = cv2.imread(cond_path)
    depth = resize_numpy_image(depth, max_resolution=opt.max_resolution)
    opt.H, opt.W = depth.shape[:2]
    if cond_type_in == 'depth':
        depth = img2tensor(depth).unsqueeze(0) / 255.
        depth = depth.to(opt.device)
    else:
        if 'depth' not in cond_preprocess:
            from ldm.modules.structure_condition.midas.api import MiDaSInference
            cond_preprocess['depth'] = MiDaSInference(model_type='dpt_hybrid').to(opt.device)
        depth = img2tensor(depth).unsqueeze(0) / 127.5 - 1.0
        depth = cond_preprocess['depth'](depth.to(opt.device)).repeat(1, 3, 1, 1)
        depth -= torch.min(depth)
        depth /= torch.max(depth)
        raise NotImplementedError

    return depth


def get_cond_inputs(opt, cond_preprocess):
    inputs = {}
    for cond_type in condition_types:
        cond_path = getattr(opt, f'{cond_type}_cond_path')
        if cond_path is None:
            continue
        cond_type_in = getattr(opt, f'{cond_type}_type_in')
        inputs[cond_type] = globals()[f'get_cond_{cond_type}'](opt, cond_path, cond_type_in, cond_preprocess)

    return inputs


def get_adapter_feature(inputs, model_ads):
    ret = None
    for cond_type in inputs.keys():
        cur_feat_list = model_ads[cond_type]['model'](inputs[cond_type])
        if ret is None:
            ret = list(map(lambda x: x * model_ads[cond_type]['weight'], cur_feat_list))
        else:
            ret = list(map(lambda x, y: x + y * model_ads[cond_type]['weight'], ret, cur_feat_list))

    return ret


def main(opt):
    assert any([opt.sketch_cond_path, opt.seg_cond_path, opt.pose_cond_path]), 'no condition was provided'
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    opt.device = device

    # SD
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, opt.ckpt, opt.ckpt_vae)
    model = model.to(device)

    # Adaptors
    model_ads = {}
    for cond_type in condition_types:
        cond_path = getattr(opt, f'{cond_type}_cond_path')
        if cond_path is None:
            continue
        model_ads[cond_type] = {}
        model_ads[cond_type]['weight'] = getattr(opt, f'{cond_type}_cond_weight')
        model_ads[cond_type]['model'] = Adapter(cin=64 if cond_type == 'sketch' else 3 * 64,
                                                channels=[320, 640, 1280, 1280][:4], nums_rb=2, ksize=1, sk=True,
                                                use_conv=False).to(device)
        model_ads[cond_type]['model'].load_state_dict(torch.load(getattr(opt, f'{cond_type}_ckpt')))

    # sampler
    if opt.sampler == 'plms':
        sampler = PLMSSampler(model)
    elif opt.sampler == 'ddim':
        sampler = DDIMSampler(model)
    else:
        raise NotImplementedError

    os.makedirs(opt.outdir, exist_ok=True)

    cond_preprocess = {}

    seed_everything(opt.seed)

    with torch.no_grad(), \
            model.ema_scope(), \
            autocast('cuda'):
        for v_idx in range(opt.n_samples):

            cond_inputs = get_cond_inputs(opt, cond_preprocess)
            features_adapter = get_adapter_feature(cond_inputs, model_ads)

            c = model.get_learned_conditioning([opt.prompt])
            if opt.scale != 1.0:
                uc = model.get_learned_conditioning([opt.neg_prompt])
            else:
                uc = None
            c, uc = fix_cond_shapes(model, c, uc)

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
                                             cond_tau=opt.cond_tau,
                                             )

            base_count = len(os.listdir(opt.outdir))
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            x_samples_ddim = x_samples_ddim.permute(0, 2, 3, 1)[0].cpu().numpy()
            x_sample = 255. * x_samples_ddim
            x_sample = Image.fromarray(x_sample.astype(np.uint8))
            x_sample.save(os.path.join(opt.outdir, f'{base_count:05}_result.png'))


if __name__ == '__main__':
    opt = parse_args()
    main(opt)
