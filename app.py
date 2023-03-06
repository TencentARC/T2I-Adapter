# demo inspired by https://huggingface.co/spaces/lambdalabs/image-mixer-demo
import argparse
import copy
import os
import shlex
import subprocess
from functools import partial
from itertools import chain

import cv2
import gradio as gr
import torch
from basicsr.utils import tensor2img
from huggingface_hub import hf_hub_url
from pytorch_lightning import seed_everything
from torch import autocast

from ldm.inference_base import (DEFAULT_NEGATIVE_PROMPT, diffusion_inference, get_adapters, get_sd_models)
from ldm.modules.extra_condition import api
from ldm.modules.extra_condition.api import (ExtraCondition, get_adapter_feature, get_cond_model)

torch.set_grad_enabled(False)

supported_cond = ['style', 'color', 'canny', 'sketch', 'openpose', 'depth']

# download the checkpoints
urls = {
    'TencentARC/T2I-Adapter': [
        'models/t2iadapter_keypose_sd14v1.pth', 'models/t2iadapter_color_sd14v1.pth',
        'models/t2iadapter_openpose_sd14v1.pth', 'models/t2iadapter_seg_sd14v1.pth',
        'models/t2iadapter_sketch_sd14v1.pth', 'models/t2iadapter_depth_sd14v1.pth',
        'third-party-models/body_pose_model.pth', "models/t2iadapter_style_sd14v1.pth",
        "models/t2iadapter_canny_sd14v1.pth", 'third-party-models/table5_pidinet.pth'
    ],
    'runwayml/stable-diffusion-v1-5': ['v1-5-pruned-emaonly.ckpt'],
    'andite/anything-v4.0': ['anything-v4.0-pruned.ckpt', 'anything-v4.0.vae.pt'],
}

if os.path.exists('models') == False:
    os.mkdir('models')
for repo in urls:
    files = urls[repo]
    for file in files:
        url = hf_hub_url(repo, file)
        name_ckp = url.split('/')[-1]
        save_path = os.path.join('models', name_ckp)
        if os.path.exists(save_path) == False:
            subprocess.run(shlex.split(f'wget {url} -O {save_path}'))

# config
parser = argparse.ArgumentParser()
parser.add_argument(
    '--sd_ckpt',
    type=str,
    default='models/v1-5-pruned-emaonly.ckpt',
    help='path to checkpoint of stable diffusion model, both .ckpt and .safetensor are supported',
)
parser.add_argument(
    '--vae_ckpt',
    type=str,
    default=None,
    help='vae checkpoint, anime SD models usually have seperate vae ckpt that need to be loaded',
)
global_opt = parser.parse_args()
global_opt.config = 'configs/stable-diffusion/sd-v1-inference.yaml'
for cond_name in supported_cond:
    setattr(global_opt, f'{cond_name}_adapter_ckpt', f'models/t2iadapter_{cond_name}_sd14v1.pth')
global_opt.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
global_opt.max_resolution = 512 * 512
global_opt.sampler = 'ddim'
global_opt.cond_weight = 1.0
global_opt.C = 4
global_opt.f = 8

# stable-diffusion model
sd_model, sampler = get_sd_models(global_opt)
# adapters and models to processing condition inputs
adapters = {}
cond_models = {}
torch.cuda.empty_cache()


def run(*args):
    with torch.inference_mode(), \
            sd_model.ema_scope(), \
            autocast('cuda'):

        inps = []
        for i in range(0, len(args) - 8, len(supported_cond)):
            inps.append(args[i:i + len(supported_cond)])

        opt = copy.deepcopy(global_opt)
        opt.prompt, opt.neg_prompt, opt.scale, opt.n_samples, opt.seed, opt.steps, opt.resize_short_edge, opt.cond_tau \
            = args[-8:]

        conds = []
        activated_conds = []

        ims1 = []
        ims2 = []
        for idx, (b, im1, im2, cond_weight) in enumerate(zip(*inps)):
            if idx > 1:
                if im1 is not None or im2 is not None:
                    if im1 is not None:
                        h, w, _ = im1.shape
                    else:
                        h, w, _ = im2.shape
                    break
        # resize all the images to the same size
        for idx, (b, im1, im2, cond_weight) in enumerate(zip(*inps)):
            if idx == 0:
                ims1.append(im1)
                ims2.append(im2)
                continue
            if im1 is not None:
                im1 = cv2.resize(im1, (w, h), interpolation=cv2.INTER_CUBIC)
            if im2 is not None:
                im2 = cv2.resize(im2, (w, h), interpolation=cv2.INTER_CUBIC)
            ims1.append(im1)
            ims2.append(im2)

        for idx, (b, _, _, cond_weight) in enumerate(zip(*inps)):
            cond_name = supported_cond[idx]
            if b == 'Nothing':
                if cond_name in adapters:
                    adapters[cond_name]['model'] = adapters[cond_name]['model'].cpu()
            else:
                activated_conds.append(cond_name)
                if cond_name in adapters:
                    adapters[cond_name]['model'] = adapters[cond_name]['model'].to(opt.device)
                else:
                    adapters[cond_name] = get_adapters(opt, getattr(ExtraCondition, cond_name))
                adapters[cond_name]['cond_weight'] = cond_weight

                process_cond_module = getattr(api, f'get_cond_{cond_name}')

                if b == 'Image':
                    if cond_name not in cond_models:
                        cond_models[cond_name] = get_cond_model(opt, getattr(ExtraCondition, cond_name))
                    conds.append(process_cond_module(opt, ims1[idx], 'image', cond_models[cond_name]))
                else:
                    conds.append(process_cond_module(opt, ims2[idx], cond_name, None))

        adapter_features, append_to_context = get_adapter_feature(
            conds, [adapters[cond_name] for cond_name in activated_conds])

        output_conds = []
        for cond in conds:
            output_conds.append(tensor2img(cond, rgb2bgr=False))

        ims = []
        seed_everything(opt.seed)
        for _ in range(opt.n_samples):
            result = diffusion_inference(opt, sd_model, sampler, adapter_features, append_to_context)
            ims.append(tensor2img(result, rgb2bgr=False))

        # Clear GPU memory cache so less likely to OOM
        torch.cuda.empty_cache()
        return ims, output_conds


def change_visible(im1, im2, val):
    outputs = {}
    if val == "Image":
        outputs[im1] = gr.update(visible=True)
        outputs[im2] = gr.update(visible=False)
    elif val == "Nothing":
        outputs[im1] = gr.update(visible=False)
        outputs[im2] = gr.update(visible=False)
    else:
        outputs[im1] = gr.update(visible=False)
        outputs[im2] = gr.update(visible=True)
    return outputs


DESCRIPTION = '# [Composable T2I-Adapter](https://github.com/TencentARC/T2I-Adapter)'

DESCRIPTION += f'<p>Gradio demo for **T2I-Adapter**: [[GitHub]](https://github.com/TencentARC/T2I-Adapter), [[Paper]](https://arxiv.org/abs/2302.08453). If T2I-Adapter is helpful, please help to ⭐ the [Github Repo](https://github.com/TencentARC/T2I-Adapter) and recommend it to your friends 😊 </p>'

DESCRIPTION += f'<p>For faster inference without waiting in queue, you may duplicate the space and upgrade to GPU in settings. <a href="https://huggingface.co/spaces/Adapter/T2I-Adapter?duplicate=true"><img style="display: inline; margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space" /></a></p>'

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)

    btns = []
    ims1 = []
    ims2 = []
    cond_weights = []

    with gr.Row():
        with gr.Column(scale=1.9):
            with gr.Box():
                gr.Markdown("<h5><center>Style & Color</center></h5>")
                with gr.Row():
                    for cond_name in supported_cond[:2]:
                        with gr.Box():
                            with gr.Column():
                                if cond_name == 'style':
                                    btn1 = gr.Radio(
                                        choices=["Image", "Nothing"],
                                        label=f"Input type for {cond_name}",
                                        interactive=True,
                                        value="Nothing",
                                    )
                                else:
                                    btn1 = gr.Radio(
                                        choices=["Image", cond_name, "Nothing"],
                                        label=f"Input type for {cond_name}",
                                        interactive=True,
                                        value="Nothing",
                                    )
                                im1 = gr.Image(
                                    source='upload', label="Image", interactive=True, visible=False, type="numpy")
                                im2 = gr.Image(
                                    source='upload', label=cond_name, interactive=True, visible=False, type="numpy")
                                cond_weight = gr.Slider(
                                    label="Condition weight",
                                    minimum=0,
                                    maximum=5,
                                    step=0.05,
                                    value=1,
                                    interactive=True)

                                fn = partial(change_visible, im1, im2)
                                btn1.change(fn=fn, inputs=[btn1], outputs=[im1, im2], queue=False)

                                btns.append(btn1)
                                ims1.append(im1)
                                ims2.append(im2)
                                cond_weights.append(cond_weight)
        with gr.Column(scale=4):
            with gr.Box():
                gr.Markdown("<h5><center>Structure</center></h5>")
                with gr.Row():
                    for cond_name in supported_cond[2:6]:
                        with gr.Box():
                            with gr.Column():
                                if cond_name == 'openpose':
                                    btn1 = gr.Radio(
                                        choices=["Image", 'pose', "Nothing"],
                                        label=f"Input type for {cond_name}",
                                        interactive=True,
                                        value="Nothing",
                                    )
                                else:
                                    btn1 = gr.Radio(
                                        choices=["Image", cond_name, "Nothing"],
                                        label=f"Input type for {cond_name}",
                                        interactive=True,
                                        value="Nothing",
                                    )
                                im1 = gr.Image(
                                    source='upload', label="Image", interactive=True, visible=False, type="numpy")
                                im2 = gr.Image(
                                    source='upload', label=cond_name, interactive=True, visible=False, type="numpy")
                                cond_weight = gr.Slider(
                                    label="Condition weight",
                                    minimum=0,
                                    maximum=5,
                                    step=0.05,
                                    value=1,
                                    interactive=True)

                                fn = partial(change_visible, im1, im2)
                                btn1.change(fn=fn, inputs=[btn1], outputs=[im1, im2], queue=False)

                                btns.append(btn1)
                                ims1.append(im1)
                                ims2.append(im2)
                                cond_weights.append(cond_weight)

    with gr.Column():
        prompt = gr.Textbox(label="Prompt")

        with gr.Accordion('Advanced options', open=False):
            neg_prompt = gr.Textbox(label="Negative Prompt", value=DEFAULT_NEGATIVE_PROMPT)
            scale = gr.Slider(
                label="Guidance Scale (Classifier free guidance)", value=7.5, minimum=1, maximum=20, step=0.1)
            n_samples = gr.Slider(label="Num samples", value=1, minimum=1, maximum=8, step=1)
            seed = gr.Slider(label="Seed", value=42, minimum=0, maximum=10000, step=1)
            steps = gr.Slider(label="Steps", value=50, minimum=10, maximum=100, step=1)
            resize_short_edge = gr.Slider(label="Image resolution", value=512, minimum=320, maximum=1024, step=1)
            cond_tau = gr.Slider(
                label="timestamp parameter that determines until which step the adapter is applied",
                value=1.0,
                minimum=0.1,
                maximum=1.0,
                step=0.05)

    with gr.Row():
        submit = gr.Button("Generate")
    output = gr.Gallery().style(grid=2, height='auto')
    cond = gr.Gallery().style(grid=2, height='auto')

    inps = list(chain(btns, ims1, ims2, cond_weights))

    inps.extend([prompt, neg_prompt, scale, n_samples, seed, steps, resize_short_edge, cond_tau])
    submit.click(fn=run, inputs=inps, outputs=[output, cond])
demo.queue().launch(debug=True, server_name='0.0.0.0')
