import copy
import gradio as gr
import torch
from basicsr.utils import tensor2img
import os
from huggingface_hub import hf_hub_url
import subprocess
import shlex
import cv2
from omegaconf import OmegaConf

from demo import create_demo_sketch, create_demo_canny, create_demo_pose
from Adapter.Sampling import diffusion_inference
from configs.utils import instantiate_from_config
from Adapter.extra_condition.api import get_cond_model, ExtraCondition
from Adapter.extra_condition import api
from Adapter.inference_base import get_base_argument_parser

torch.set_grad_enabled(False)

urls = {
    'TencentARC/T2I-Adapter':[
        'models_XL/adapter-xl-canny.pth', 'models_XL/adapter-xl-sketch.pth',
        'models_XL/adapter-xl-openpose.pth', 'third-party-models/body_pose_model.pth',
        'third-party-models/table5_pidinet.pth'
    ]
}

if os.path.exists('checkpoints') == False:
    os.mkdir('checkpoints')
for repo in urls:
    files = urls[repo]
    for file in files:
        url = hf_hub_url(repo, file)
        name_ckp = url.split('/')[-1]
        save_path = os.path.join('checkpoints',name_ckp)
        if os.path.exists(save_path) == False:
            subprocess.run(shlex.split(f'wget {url} -O {save_path}'))

parser = get_base_argument_parser()
global_opt = parser.parse_args()
global_opt.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

DESCRIPTION = '# [T2I-Adapter-XL](https://github.com/TencentARC/T2I-Adapter)'

DESCRIPTION += f'<p>Gradio demo for **T2I-Adapter-XL**: [[GitHub]](https://github.com/TencentARC/T2I-Adapter). If T2I-Adapter-XL is helpful, please help to ⭐ the [Github Repo](https://github.com/TencentARC/T2I-Adapter) and recommend it to your friends 😊 </p>'

# DESCRIPTION += f'<p>For faster inference without waiting in queue, you may duplicate the space and upgrade to GPU in settings. <a href="https://huggingface.co/spaces/Adapter/T2I-Adapter?duplicate=true"><img style="display: inline; margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space" /></a></p>'

# diffusion sampler creation
sampler = diffusion_inference('stabilityai/stable-diffusion-xl-base-1.0')

def run(input_image, in_type, prompt, a_prompt, n_prompt, ddim_steps, scale, seed, cond_name, con_strength):
    in_type = in_type.lower()
    prompt = prompt+', '+a_prompt
    config = OmegaConf.load(f'configs/inference/Adapter-XL-{cond_name}.yaml')
    # Adapter creation
    adapter_config = config.model.params.adapter_config
    adapter = instantiate_from_config(adapter_config).cuda()
    adapter.load_state_dict(torch.load(config.model.params.adapter_config.pretrained))
    cond_model = get_cond_model(global_opt, getattr(ExtraCondition, cond_name))
    process_cond_module = getattr(api, f'get_cond_{cond_name}')

    # diffusion generation
    cond = process_cond_module(
        global_opt,
        input_image, 
        cond_inp_type = in_type, 
        cond_model = cond_model
    )
    with torch.no_grad():
        adapter_features = adapter(cond)

        for i in range(len(adapter_features)):
            adapter_features[i] = adapter_features[i]*con_strength

        result = sampler.inference(
            prompt = prompt, 
            prompt_n = n_prompt,
            steps = ddim_steps,
            adapter_features = copy.deepcopy(adapter_features), 
            guidance_scale = scale,
            size = (cond.shape[-2], cond.shape[-1]),
            seed= seed,
        )
    im_cond = tensor2img(cond)

    return result[:,:,::-1], im_cond

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Tabs():
        with gr.TabItem('Sketch guided'):
            create_demo_sketch(run)
        with gr.TabItem('Canny guided'):
            create_demo_canny(run)
        with gr.TabItem('Keypoint guided'):
            create_demo_pose(run)

demo.queue(concurrency_count=3, max_size=20)
demo.launch(server_name="0.0.0.0")
