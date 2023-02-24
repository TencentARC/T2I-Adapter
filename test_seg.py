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

torch.set_grad_enabled(False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/test-sketch"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="A black Honda motorcycle parked in front of a garage"
    )
    parser.add_argument(
        "--neg_prompt",
        type=str,
        default="ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face"
    )
    parser.add_argument(
        "--path_cond",
        type=str,
        default="examples/seg/motor.png"
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
        default="models/t2iadapter_seg_sd14v1.pth"
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
        default=1.0,
        help='timestamp parameter that determines until which step the adapter is applied, similar as Prompt-to-Prompt tau'
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
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
    model_ad = Adapter(cin=int(3*64), channels=[320, 640, 1280, 1280][:4], nums_rb=2, ksize=1, sk=True, use_conv=False).to(device)
    model_ad.load_state_dict(torch.load(opt.ckpt_ad))

    # sampler
    if opt.sampler == 'plms':
        sampler = PLMSSampler(model)
    elif opt.sampler == 'ddim':
        sampler = DDIMSampler(model)
    else:
        raise NotImplementedError

    os.makedirs(opt.outdir, exist_ok=True)

    seed_everything(opt.seed)

    with torch.inference_mode(), \
            model.ema_scope(), \
            autocast('cuda'):
        for v_idx in range(opt.n_samples):
            # costumer input
            mask = cv2.imread(opt.path_cond)
            mask = resize_numpy_image(mask, max_resolution=opt.max_resolution)
            opt.H, opt.W = mask.shape[:2]
            mask = img2tensor(mask, bgr2rgb=True, float32=True) / 255.
            mask = mask.unsqueeze(0)

            c = model.get_learned_conditioning([opt.prompt])
            if opt.scale != 1.0:
                uc = model.get_learned_conditioning([opt.neg_prompt])
            else:
                uc = None
            c, uc = fix_cond_shapes(model, c, uc)

            base_count = len(os.listdir(opt.outdir))

            features_adapter = model_ad(mask.to(device))

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
