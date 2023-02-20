from demo.model import Model_all
import gradio as gr
from demo.demos import create_demo_keypose, create_demo_sketch, create_demo_draw
import torch

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

demo.queue(api_open=False).launch(server_name='9.134.172.88')