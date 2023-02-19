import os
import os.path as osp

import cv2
import numpy as np
import torch
from basicsr.utils import img2tensor, tensor2img
from pytorch_lightning import seed_everything
from ldm.models.diffusion.plms import PLMSSampler
from ldm.modules.encoders.adapter import Adapter
from ldm.util import instantiate_from_config
from model_edge import pidinet
import gradio as gr
from omegaconf import OmegaConf


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
    # if len(m) > 0 and verbose:
    #     print("missing keys:")
    #     print(m)
    # if len(u) > 0 and verbose:
    #     print("unexpected keys:")
    #     print(u)

    model.cuda()
    model.eval()
    return model

device = 'cuda'
config = OmegaConf.load("configs/stable-diffusion/test_sketch.yaml")
config.model.params.cond_stage_config.params.device = device
model = load_model_from_config(config, "models/sd-v1-4.ckpt").to(device)
current_base = 'sd-v1-4.ckpt'
model_ad = Adapter(channels=[320, 640, 1280, 1280][:4], nums_rb=2, ksize=1, sk=True, use_conv=False).to(device)
model_ad.load_state_dict(torch.load("models/t2iadapter_sketch_sd14v1.pth"))
net_G = pidinet()
ckp = torch.load('models/table5_pidinet.pth', map_location='cpu')['state_dict']
net_G.load_state_dict({k.replace('module.',''):v for k, v in ckp.items()})
net_G.to(device)
sampler = PLMSSampler(model)
save_memory=True


def process(input_img, type_in, color_back, prompt, neg_prompt, fix_sample, scale, con_strength, base_model):
    global current_base
    if current_base != base_model:
        ckpt = os.path.join("models", base_model)
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        else:
            sd = pl_sd
        model.load_state_dict(sd, strict=False) #load_model_from_config(config, os.path.join("models", base_model)).to(device)
        current_base = base_model
    con_strength = int((1-con_strength)*50)
    if fix_sample == 'True':
        seed_everything(42)
    im = cv2.resize(input_img,(512,512))

    if type_in == 'Sketch':
        # net_G = net_G.cpu()
        if color_back == 'White':
            im = 255-im
        im_edge = im.copy()
        im = img2tensor(im)[0].unsqueeze(0).unsqueeze(0)/255.
        # edge = 1-edge # for white background
        im = im>0.5
        im = im.float()
    elif type_in == 'Image':
        im = img2tensor(im).unsqueeze(0)/255.
        im = net_G(im.to(device))[-1]
        im = im>0.5
        im = im.float()
        im_edge = tensor2img(im)
    
    with torch.no_grad():
        c = model.get_learned_conditioning([prompt])
        nc = model.get_learned_conditioning([neg_prompt])
        # extract condition features
        features_adapter = model_ad(im.to(device))
        shape = [4, 64, 64]

        # sampling
        samples_ddim, _ = sampler.sample(S=50,
                                        conditioning=c,
                                        batch_size=1,
                                        shape=shape,
                                        verbose=False,
                                        unconditional_guidance_scale=scale,
                                        unconditional_conditioning=nc,
                                        eta=0.0,
                                        x_T=None,
                                        features_adapter1=features_adapter,
                                        mode = 'sketch',
                                        con_strength = con_strength)

        x_samples_ddim = model.decode_first_stage(samples_ddim)
        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        x_samples_ddim = x_samples_ddim.permute(0, 2, 3, 1).numpy()[0]
        x_samples_ddim = 255.*x_samples_ddim
        x_samples_ddim = x_samples_ddim.astype(np.uint8)

    return [im_edge, x_samples_ddim]

DESCRIPTION = '''# T2I-Adapter (Sketch)
[Paper](https://arxiv.org/abs/2302.08453)               [GitHub](https://github.com/TencentARC/T2I-Adapter) 

This gradio demo is for sketch-guided generation. The current functions include:
- Sketch to Image Generation
- Image to Image Generation 
- Generation with **Anything** setting
'''
block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(source='upload', type="numpy")
            prompt = gr.Textbox(label="Prompt")
            neg_prompt = gr.Textbox(label="Negative Prompt",
            value='ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face')
            with gr.Row():
                type_in = gr.inputs.Radio(['Sketch', 'Image'], type="value", default='Image', label='Input Types\n (You can input an image or a sketch)')
                color_back = gr.inputs.Radio(['White', 'Black'], type="value", default='Black', label='Color of the sketch background\n (Only work for sketch input)')
            run_button = gr.Button(label="Run")
            con_strength = gr.Slider(label="Controling Strength (The guidance strength of the sketch to the result)", minimum=0, maximum=1, value=0.4, step=0.1)
            scale = gr.Slider(label="Guidance Scale (Classifier free guidance)", minimum=0.1, maximum=30.0, value=7.5, step=0.1)
            fix_sample = gr.inputs.Radio(['True', 'False'], type="value", default='False', label='Fix Sampling\n (Fix the random seed)')
            base_model = gr.inputs.Radio(['sd-v1-4.ckpt', 'anything-v4.0-pruned.ckpt'], type="value", default='sd-v1-4.ckpt', label='The base model you want to use')
        with gr.Column():
            result = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
        ips = [input_img, type_in, color_back, prompt, neg_prompt, fix_sample, scale, con_strength, base_model]
    run_button.click(fn=process, inputs=ips, outputs=[result])

block.launch(server_name='9.134.172.88')
                
