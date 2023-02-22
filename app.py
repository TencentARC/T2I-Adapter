import os
# os.system('pip3 install openmim')
os.system('mim install mmcv-full==1.7.0')
# os.system('pip3 install mmpose')
# os.system('pip3 install mmdet')
# os.system('pip3 install gradio==3.19.1')
#os.system('pip3 install psutil')

from demo.model import Model_all
import gradio as gr
from demo.demos import create_demo_keypose, create_demo_sketch, create_demo_draw
import torch
import subprocess
import shlex
from huggingface_hub import hf_hub_url

urls = {
    'TencentARC/T2I-Adapter':['models/t2iadapter_keypose_sd14v1.pth', 'models/t2iadapter_seg_sd14v1.pth', 'models/t2iadapter_sketch_sd14v1.pth'],
    'CompVis/stable-diffusion-v-1-4-original':['sd-v1-4.ckpt'],
    'andite/anything-v4.0':['anything-v4.0-pruned.ckpt', 'anything-v4.0.vae.pt'],
}
urls_mmpose = [
    'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth',
    'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth',
]
if os.path.exists('models') == False:
    os.mkdir('models')
for repo in urls:
    files = urls[repo]
    for file in files:
        url = hf_hub_url(repo, file)
        name_ckp = url.split('/')[-1]
        save_path = os.path.join('models',name_ckp)
        if os.path.exists(save_path) == False:
            subprocess.run(shlex.split(f'wget {url} -O {save_path}'))

for url in urls_mmpose:
    name_ckp = url.split('/')[-1]
    save_path = os.path.join('models',name_ckp)
    if os.path.exists(save_path) == False:
        subprocess.run(shlex.split(f'wget {url} -O {save_path}'))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Model_all(device)

DESCRIPTION = '''# T2I-Adapter (Sketch & Keypose)
[Paper](https://arxiv.org/abs/2302.08453)               [GitHub](https://github.com/TencentARC/T2I-Adapter) 

This gradio demo is for a simple experience of T2I-Adapter:
- Keypose/Sketch to Image Generation
- Image to Image Generation 
- Support the base model of Stable Diffusion v1.4 and Anything 4.0
'''

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Tabs():
        with gr.TabItem('Keypose'):
            create_demo_keypose(model.process_keypose)
        with gr.TabItem('Sketch'):
            create_demo_sketch(model.process_sketch)
        with gr.TabItem('Draw'):
            create_demo_draw(model.process_draw)

# demo.queue(api_open=False).launch(server_name='0.0.0.0')
# demo.queue(show_api=False, enable_queue=False).launch(server_name='0.0.0.0')
demo.queue().launch(debug=True, server_name='0.0.0.0')