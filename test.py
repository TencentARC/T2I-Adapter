from omegaconf import OmegaConf
import torch
import os
import cv2
import datetime
from huggingface_hub import hf_hub_url
import subprocess
import shlex
import copy
from basicsr.utils import tensor2img

from Adapter.Sampling import diffusion_inference
from configs.utils import instantiate_from_config
from Adapter.inference_base import get_base_argument_parser
from Adapter.extra_condition.api import get_cond_model, ExtraCondition
from Adapter.extra_condition import api

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

# config
parser = get_base_argument_parser()
parser.add_argument(
    '--model_id',
    type=str,
    default="stabilityai/stable-diffusion-xl-base-1.0",
    help='huggingface url to stable diffusion model',
)
parser.add_argument(
    '--config',
    type=str,
    default='configs/inference/Adapter-XL-sketch.yaml',
    help='config path to T2I-Adapter',
)
parser.add_argument(
    '--path_source',
    type=str,
    default='examples/dog.png',
    help='config path to the source image',
)
parser.add_argument(
    '--in_type',
    type=str,
    default='image',
    help='config path to the source image',
)
global_opt = parser.parse_args()
global_opt.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if __name__ == '__main__':
    config = OmegaConf.load(global_opt.config)
    # Adapter creation
    cond_name = config.model.params.adapter_config.name
    adapter_config = config.model.params.adapter_config
    adapter = instantiate_from_config(adapter_config).cuda()
    adapter.load_state_dict(torch.load(config.model.params.adapter_config.pretrained))
    cond_model = get_cond_model(global_opt, getattr(ExtraCondition, cond_name))
    process_cond_module = getattr(api, f'get_cond_{cond_name}')

    # diffusion sampler creation
    sampler = diffusion_inference(global_opt.model_id)
    
    # diffusion generation
    cond = process_cond_module(
        global_opt,
        global_opt.path_source, 
        cond_inp_type = global_opt.in_type, 
        cond_model = cond_model
    )
    with torch.no_grad():
        adapter_features = adapter(cond)
        result = sampler.inference(
            prompt = global_opt.prompt, 
            prompt_n = global_opt.neg_prompt,
            steps = global_opt.steps,
            adapter_features = copy.deepcopy(adapter_features), 
            guidance_scale = global_opt.scale,
            size = (cond.shape[-2], cond.shape[-1]),
            seed= global_opt.seed,
        )

    # save results
    root_results = os.path.join('results', cond_name)
    if not os.path.exists(root_results):
        os.makedirs(root_results)
    now = datetime.datetime.now()
    formatted_date = now.strftime("%Y-%m-%d")
    formatted_time = now.strftime("%H:%M:%S")
    im_cond = tensor2img(cond)
    cv2.imwrite(os.path.join(root_results, formatted_date+'-'+formatted_time+'_image.png'), result)
    cv2.imwrite(os.path.join(root_results, formatted_date+'-'+formatted_time+'_condition.png'), im_cond)
