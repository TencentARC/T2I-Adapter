import gradio as gr
import numpy as np

def create_map():
    return np.zeros(shape=(512, 1024), dtype=np.uint8)+255


def create_demo_keypose(process):
    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown('T2I-Adapter (Keypose)')
        with gr.Row():
            with gr.Column():
                input_img = gr.Image(source='upload', type="numpy")
                prompt = gr.Textbox(label="Prompt")
                neg_prompt = gr.Textbox(label="Negative Prompt",
                value='ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face')
                pos_prompt = gr.Textbox(label="Positive Prompt",
                value = 'crafted, elegant, meticulous, magnificent, maximum details, extremely hyper aesthetic, intricately detailed')
                with gr.Row():
                    type_in = gr.inputs.Radio(['Keypose', 'Image'], type="value", default='Image', label='Input Types\n (You can input an image or a keypose map)')
                    fix_sample = gr.inputs.Radio(['True', 'False'], type="value", default='False', label='Fix Sampling\n (Fix the random seed to produce a fixed output)')
                run_button = gr.Button(label="Run")
                con_strength = gr.Slider(label="Controling Strength (The guidance strength of the keypose to the result)", minimum=0, maximum=1, value=1, step=0.1)
                scale = gr.Slider(label="Guidance Scale (Classifier free guidance)", minimum=0.1, maximum=30.0, value=9, step=0.1)
                base_model = gr.inputs.Radio(['sd-v1-4.ckpt', 'anything-v4.0-pruned.ckpt'], type="value", default='sd-v1-4.ckpt', label='The base model you want to use')
            with gr.Column():
                result = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
        ips = [input_img, type_in, prompt, neg_prompt, pos_prompt, fix_sample, scale, con_strength, base_model]
        run_button.click(fn=process, inputs=ips, outputs=[result])
    return demo

def create_demo_sketch(process):
    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown('T2I-Adapter (Sketch)')
        with gr.Row():
            with gr.Column():
                input_img = gr.Image(source='upload', type="numpy")
                prompt = gr.Textbox(label="Prompt")
                neg_prompt = gr.Textbox(label="Negative Prompt",
                value='ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face')
                pos_prompt = gr.Textbox(label="Positive Prompt",
                value = 'crafted, elegant, meticulous, magnificent, maximum details, extremely hyper aesthetic, intricately detailed')
                with gr.Row():
                    type_in = gr.inputs.Radio(['Sketch', 'Image'], type="value", default='Image', label='Input Types\n (You can input an image or a sketch)')
                    color_back = gr.inputs.Radio(['White', 'Black'], type="value", default='Black', label='Color of the sketch background\n (Only work for sketch input)')
                run_button = gr.Button(label="Run")
                con_strength = gr.Slider(label="Controling Strength (The guidance strength of the sketch to the result)", minimum=0, maximum=1, value=0.4, step=0.1)
                scale = gr.Slider(label="Guidance Scale (Classifier free guidance)", minimum=0.1, maximum=30.0, value=9, step=0.1)
                fix_sample = gr.inputs.Radio(['True', 'False'], type="value", default='False', label='Fix Sampling\n (Fix the random seed)')
                base_model = gr.inputs.Radio(['sd-v1-4.ckpt', 'anything-v4.0-pruned.ckpt'], type="value", default='sd-v1-4.ckpt', label='The base model you want to use')
            with gr.Column():
                result = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
            ips = [input_img, type_in, color_back, prompt, neg_prompt, pos_prompt, fix_sample, scale, con_strength, base_model]
        run_button.click(fn=process, inputs=ips, outputs=[result])
    return demo

def create_demo_draw(process):
    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown('T2I-Adapter (Hand-free drawing)')
        with gr.Row():
            with gr.Column():
                create_button = gr.Button(label="Start", value='Hand-free drawing')
                input_img = gr.Image(source='upload', type="numpy",tool='sketch')
                create_button.click(fn=create_map, outputs=[input_img])
                prompt = gr.Textbox(label="Prompt")
                neg_prompt = gr.Textbox(label="Negative Prompt",
                value='ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face')
                pos_prompt = gr.Textbox(label="Positive Prompt",
                value = 'crafted, elegant, meticulous, magnificent, maximum details, extremely hyper aesthetic, intricately detailed')
                run_button = gr.Button(label="Run")
                con_strength = gr.Slider(label="Controling Strength (The guidance strength of the sketch to the result)", minimum=0, maximum=1, value=0.4, step=0.1)
                scale = gr.Slider(label="Guidance Scale (Classifier free guidance)", minimum=0.1, maximum=30.0, value=9, step=0.1)
                fix_sample = gr.inputs.Radio(['True', 'False'], type="value", default='False', label='Fix Sampling\n (Fix the random seed)')
                base_model = gr.inputs.Radio(['sd-v1-4.ckpt', 'anything-v4.0-pruned.ckpt'], type="value", default='sd-v1-4.ckpt', label='The base model you want to use')
            with gr.Column():
                result = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
            ips = [input_img, prompt, neg_prompt, pos_prompt, fix_sample, scale, con_strength, base_model]
        run_button.click(fn=process, inputs=ips, outputs=[result])
    return demo