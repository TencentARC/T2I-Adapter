import gradio as gr

def create_demo_sketch(run):
    cond_name = gr.State(value='sketch')
    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown('## Control Stable Diffusion-XL with Sketch Maps')
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(source='upload', type='numpy')
                prompt = gr.Textbox(label='Prompt')
                run_button = gr.Button(label='Run')
                in_type = gr.Radio(
                            choices=["Image", "Sketch"],
                            label=f"Input type for Sketch",
                            interactive=True,
                            value="Image",
                        )
                with gr.Accordion('Advanced options', open=False):
                    con_strength = gr.Slider(label='Control Strength',
                                      minimum=0.0,
                                      maximum=1.0,
                                      value=1.0,
                                      step=0.1)
                    ddim_steps = gr.Slider(label='Steps',
                                           minimum=1,
                                           maximum=100,
                                           value=20,
                                           step=1)
                    scale = gr.Slider(label='Guidance Scale',
                                      minimum=0.1,
                                      maximum=30.0,
                                      value=7.5,
                                      step=0.1)
                    seed = gr.Slider(label='Seed',
                                     minimum=-1,
                                     maximum=2147483647,
                                     step=1,
                                     randomize=True)
                    a_prompt = gr.Textbox(
                        label='Added Prompt',
                        value='in real world, high quality')
                    n_prompt = gr.Textbox(
                        label='Negative Prompt',
                        value='extra digit, fewer digits, cropped, worst quality, low quality'
                    )
            with gr.Column():
                result_gallery = gr.Gallery(label='Output',
                                            show_label=False,
                                            elem_id='gallery').style(
                                                grid=2, height='auto')
        ips = [
            input_image, in_type, prompt, a_prompt, n_prompt,
            ddim_steps, scale, seed, cond_name, con_strength
        ]
        run_button.click(fn=run,
                         inputs=ips,
                         outputs=[result_gallery],
                         api_name='sketch')
    return demo

def create_demo_canny(run):
    cond_name = gr.State(value='canny')
    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown('## Control Stable Diffusion-XL with Canny Maps')
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(source='upload', type='numpy')
                prompt = gr.Textbox(label='Prompt')
                run_button = gr.Button(label='Run')
                in_type = gr.Radio(
                            choices=["Image", "Canny"],
                            label=f"Input type for Canny",
                            interactive=True,
                            value="Image",
                        )
                with gr.Accordion('Advanced options', open=False):
                    con_strength = gr.Slider(label='Control Strength',
                                      minimum=0.0,
                                      maximum=1.0,
                                      value=1.0,
                                      step=0.1)
                    ddim_steps = gr.Slider(label='Steps',
                                           minimum=1,
                                           maximum=100,
                                           value=20,
                                           step=1)
                    scale = gr.Slider(label='Guidance Scale',
                                      minimum=0.1,
                                      maximum=30.0,
                                      value=7.5,
                                      step=0.1)
                    seed = gr.Slider(label='Seed',
                                     minimum=-1,
                                     maximum=2147483647,
                                     step=1,
                                     randomize=True)
                    a_prompt = gr.Textbox(
                        label='Added Prompt',
                        value='in real world, high quality')
                    n_prompt = gr.Textbox(
                        label='Negative Prompt',
                        value='extra digit, fewer digits, cropped, worst quality, low quality'
                    )
            with gr.Column():
                result_gallery = gr.Gallery(label='Output',
                                            show_label=False,
                                            elem_id='gallery').style(
                                                grid=2, height='auto')
        ips = [
            input_image, in_type, prompt, a_prompt, n_prompt,
            ddim_steps, scale, seed, cond_name, con_strength
        ]
        run_button.click(fn=run,
                         inputs=ips,
                         outputs=[result_gallery],
                         api_name='canny')
    return demo

def create_demo_pose(run):
    cond_name = gr.State(value='openpose')
    in_type = gr.State(value='Image')
    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown('## Control Stable Diffusion-XL with Keypoint Maps')
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(source='upload', type='numpy')
                prompt = gr.Textbox(label='Prompt')
                run_button = gr.Button(label='Run')
                with gr.Accordion('Advanced options', open=False):
                    con_strength = gr.Slider(label='Control Strength',
                                      minimum=0.0,
                                      maximum=1.0,
                                      value=1.0,
                                      step=0.1)
                    ddim_steps = gr.Slider(label='Steps',
                                           minimum=1,
                                           maximum=100,
                                           value=20,
                                           step=1)
                    scale = gr.Slider(label='Guidance Scale',
                                      minimum=0.1,
                                      maximum=30.0,
                                      value=7.5,
                                      step=0.1)
                    seed = gr.Slider(label='Seed',
                                     minimum=-1,
                                     maximum=2147483647,
                                     step=1,
                                     randomize=True)
                    a_prompt = gr.Textbox(
                        label='Added Prompt',
                        value='in real world, high quality')
                    n_prompt = gr.Textbox(
                        label='Negative Prompt',
                        value='extra digit, fewer digits, cropped, worst quality, low quality'
                    )
            with gr.Column():
                result_gallery = gr.Gallery(label='Output',
                                            show_label=False,
                                            elem_id='gallery').style(
                                                grid=2, height='auto')
        ips = [
            input_image, in_type, prompt, a_prompt, n_prompt,
            ddim_steps, scale, seed, cond_name, con_strength
        ]
        run_button.click(fn=run,
                         inputs=ips,
                         outputs=[result_gallery],
                         api_name='openpose')
    return demo