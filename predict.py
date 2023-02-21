from typing import List
import numpy as np
import cv2
import torch
from omegaconf import OmegaConf
from cog import BasePredictor, Input, Path
from basicsr.utils import tensor2img, img2tensor

from ldm.modules.encoders.adapter import Adapter
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config
from model_edge import pidinet


MODEL_CACHE = "models"


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

        device = "cuda"
        config_file = "configs/stable-diffusion/test_sketch.yaml"  # test_sketch.yaml and test_keypose.yaml are same
        config = OmegaConf.load(config_file)

        # stable diffusion models
        self.models = {
            k: load_model_from_config(config, f"models/{k}.ckpt").to(device)
            for k in ["sd-v1-4", "anything-v4.0-pruned"]
        }

        # Adaptor
        self.model_edge_ad = Adapter(
            channels=[320, 640, 1280, 1280][:4],
            nums_rb=2,
            ksize=1,
            sk=True,
            use_conv=False,
        ).to(device)
        self.model_edge_ad.load_state_dict(
            torch.load("models/t2iadapter_sketch_sd14v1.pth")
        )

        self.model_pose_ad = Adapter(
            cin=int(3 * 64),
            channels=[320, 640, 1280, 1280][:4],
            nums_rb=2,
            ksize=1,
            sk=True,
            use_conv=False,
        ).to(device)
        self.model_pose_ad.load_state_dict(
            torch.load("models/t2iadapter_keypose_sd14v1.pth")
        )

        # edge_generator
        self.net_G = pidinet()
        ckp = torch.load("models/table5_pidinet.pth", map_location="cpu")["state_dict"]
        self.net_G.load_state_dict(
            {k.replace("module.", ""): v for k, v in ckp.items()}
        )
        self.net_G.to(device)

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="A car with flying wings",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default="ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face",
        ),
        image: Path = Input(
            description="Input image",
        ),
        model_checkpoint: str = Input(
            description="Choose a model.",
            choices=["sd-v1-4", "anything-v4.0-pruned"],
            default="sd-v1-4",
        ),
        type_in: str = Input(
            description="Choose type of your input. When image is chosen, output will be the extracted sketch and the generated images.",
            choices=["sketch", "image", "keypose"],
            default="image",
        ),
        plms: bool = Input(
            description="Use plms sampling if set to True.",
            default=True,
        ),
        dpm_solver: bool = Input(
            description="Use se dpm_solver sampling if set to True.",
            default=False,
        ),
        width: int = Input(
            description="Width of output image. Lower the width if out of memory.",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=512,
        ),
        height: int = Input(
            description="Height of output image. Lower the height if out of memory.",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=512,
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""

        model = self.models[model_checkpoint]

        output_paths = []
        with torch.no_grad():
            for v_idx in range(num_outputs):
                if dpm_solver:
                    sampler = DPMSolverSampler(model)
                elif plms:
                    sampler = PLMSSampler(model)
                else:
                    sampler = DDIMSampler(model)
                c = model.get_learned_conditioning([prompt])

                if type_in == "sketch":
                    # costumer input
                    edge = cv2.imread(str(image))
                    edge = cv2.resize(edge, (width, height))
                    edge = img2tensor(edge)[0].unsqueeze(0).unsqueeze(0) / 255.0

                    # edge = 1-edge # for white background
                    edge = edge > 0.5
                    edge = edge.float()
                elif type_in == "image":
                    im = cv2.imread(str(image))
                    im = cv2.resize(im, (width, height))
                    im = img2tensor(im).unsqueeze(0) / 255.0
                    edge = self.net_G(im.cuda(non_blocking=True))[-1]

                    edge = edge > 0.5
                    edge = edge.float()
                    im_edge = tensor2img(edge)
                    if len(output_paths) == 0:
                        out_path = f"/tmp/sketch.png"
                        cv2.imwrite(out_path, im_edge)
                        output_paths.append(Path(out_path))

                elif type_in == "keypose":
                    pose = cv2.imread(str(image))
                    pose = cv2.resize(pose, (width, height))
                    pose = img2tensor(pose, bgr2rgb=True, float32=True) / 255.0
                    pose = pose.unsqueeze(0)

                else:
                    raise TypeError("Wrong input condition.")

                features_adapter = (
                    self.model_pose_ad(pose.to("cuda"))
                    if type_in == "keypose"
                    else self.model_edge_ad(edge.to("cuda"))
                )

                downsampling_factor = 8
                latent_channels = 4
                shape = [
                    latent_channels,
                    height // downsampling_factor,
                    width // downsampling_factor,
                ]
                ddim_eta = 0.0
                samples_ddim, intermediates = sampler.sample(
                    S=num_inference_steps,
                    conditioning=c,
                    batch_size=1,
                    shape=shape,
                    verbose=False,
                    unconditional_guidance_scale=guidance_scale,
                    unconditional_conditioning=model.get_learned_conditioning(
                        [negative_prompt]
                    ),
                    eta=ddim_eta,
                    x_T=None,
                    features_adapter1=features_adapter,
                    mode=type_in,
                )

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp(
                    (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0
                )
                x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                out_path = f"/tmp/out_{v_idx}.png"
                for id_sample, x_sample in enumerate(x_samples_ddim):
                    x_sample = 255.0 * x_sample
                    img = x_sample.astype(np.uint8)
                    cv2.imwrite(out_path, img[:, :, ::-1])
                    output_paths.append(Path(out_path))

        return output_paths


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    if "state_dict" in pl_sd:
        sd = pl_sd["state_dict"]
    else:
        sd = pl_sd
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
