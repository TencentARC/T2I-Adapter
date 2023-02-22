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
import mmcv
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import (inference_top_down_pose_model, init_pose_model, process_mmdet_results, vis_pose_result)

skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10],
            [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]

pose_kpt_color = [[51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255], [0, 255, 0],
                  [255, 128, 0], [0, 255, 0], [255, 128, 0], [0, 255, 0], [255, 128, 0], [0, 255, 0], [255, 128, 0],
                  [0, 255, 0], [255, 128, 0], [0, 255, 0], [255, 128, 0]]

pose_link_color = [[0, 255, 0], [0, 255, 0], [255, 128, 0], [255, 128, 0],
                   [51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255], [0, 255, 0], [255, 128, 0],
                   [0, 255, 0], [255, 128, 0], [51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255],
                   [51, 153, 255], [51, 153, 255], [51, 153, 255]]

def imshow_keypoints(img,
                     pose_result,
                     skeleton=None,
                     kpt_score_thr=0.1,
                     pose_kpt_color=None,
                     pose_link_color=None,
                     radius=4,
                     thickness=1):
    """Draw keypoints and links on an image.

    Args:
            img (ndarry): The image to draw poses on.
            pose_result (list[kpts]): The poses to draw. Each element kpts is
                a set of K keypoints as an Kx3 numpy.ndarray, where each
                keypoint is represented as x, y, score.
            kpt_score_thr (float, optional): Minimum score of keypoints
                to be shown. Default: 0.3.
            pose_kpt_color (np.array[Nx3]`): Color of N keypoints. If None,
                the keypoint will not be drawn.
            pose_link_color (np.array[Mx3]): Color of M links. If None, the
                links will not be drawn.
            thickness (int): Thickness of lines.
    """

    img_h, img_w, _ = img.shape
    img = np.zeros(img.shape)

    for idx, kpts in enumerate(pose_result):
        if idx > 1:
            continue
        kpts = kpts['keypoints']
        # print(kpts)
        kpts = np.array(kpts, copy=False)

        # draw each point on image
        if pose_kpt_color is not None:
            assert len(pose_kpt_color) == len(kpts)

            for kid, kpt in enumerate(kpts):
                x_coord, y_coord, kpt_score = int(kpt[0]), int(kpt[1]), kpt[2]

                if kpt_score < kpt_score_thr or pose_kpt_color[kid] is None:
                    # skip the point that should not be drawn
                    continue

                color = tuple(int(c) for c in pose_kpt_color[kid])
                cv2.circle(img, (int(x_coord), int(y_coord)), radius, color, -1)

        # draw links
        if skeleton is not None and pose_link_color is not None:
            assert len(pose_link_color) == len(skeleton)

            for sk_id, sk in enumerate(skeleton):
                pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
                pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))

                if (pos1[0] <= 0 or pos1[0] >= img_w or pos1[1] <= 0 or pos1[1] >= img_h or pos2[0] <= 0
                        or pos2[0] >= img_w or pos2[1] <= 0 or pos2[1] >= img_h or kpts[sk[0], 2] < kpt_score_thr
                        or kpts[sk[1], 2] < kpt_score_thr or pose_link_color[sk_id] is None):
                    # skip the link that should not be drawn
                    continue
                color = tuple(int(c) for c in pose_link_color[sk_id])
                cv2.line(img, pos1, pos2, color, thickness=thickness)

    return img

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

    model.cuda()
    model.eval()
    return model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = OmegaConf.load("configs/stable-diffusion/test_keypose.yaml")
config.model.params.cond_stage_config.params.device = device
model = load_model_from_config(config, "models/sd-v1-4.ckpt").to(device)
current_base = 'sd-v1-4.ckpt'
model_ad = Adapter(cin=int(3*64), channels=[320, 640, 1280, 1280][:4], nums_rb=2, ksize=1, sk=True, use_conv=False).to(device)
model_ad.load_state_dict(torch.load("models/t2iadapter_keypose_sd14v1.pth"))
sampler = PLMSSampler(model)
## mmpose
det_config = 'models/faster_rcnn_r50_fpn_coco.py' 
det_checkpoint = 'models/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
pose_config = 'models/hrnet_w48_coco_256x192.py'
pose_checkpoint = 'models/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
det_cat_id = 1
bbox_thr = 0.2
## detector
det_config_mmcv = mmcv.Config.fromfile(det_config)
det_model = init_detector(det_config_mmcv, det_checkpoint, device=device)
pose_config_mmcv = mmcv.Config.fromfile(pose_config)
pose_model = init_pose_model(pose_config_mmcv, pose_checkpoint, device=device)
W, H = 512, 512


def process(input_img, type_in, prompt, neg_prompt, pos_prompt, fix_sample, scale, con_strength, base_model):
    global current_base
    if current_base != base_model:
        ckpt = os.path.join("models", base_model)
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        else:
            sd = pl_sd
        model.load_state_dict(sd, strict=False)
        current_base = base_model
    con_strength = int((1-con_strength)*50)
    if fix_sample == 'True':
        seed_everything(42)
    im = cv2.resize(input_img,(W,H))

    if type_in == 'Keypose':
        im_pose = im.copy()
        im = img2tensor(im).unsqueeze(0)/255.
    elif type_in == 'Image':
        image = im.copy()
        im = img2tensor(im).unsqueeze(0)/255.
        mmdet_results = inference_detector(det_model, image)
        # keep the person class bounding boxes.
        person_results = process_mmdet_results(mmdet_results, det_cat_id)

        # optional
        return_heatmap = False
        dataset = pose_model.cfg.data['test']['type']

        # e.g. use ('backbone', ) to return backbone feature
        output_layer_names = None
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            image,
            person_results,
            bbox_thr=bbox_thr,
            format='xyxy',
            dataset=dataset,
            dataset_info=None,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        # show the results
        im_pose = imshow_keypoints(
            image,
            pose_results,
            skeleton=skeleton,
            pose_kpt_color=pose_kpt_color,
            pose_link_color=pose_link_color,
            radius=2,
            thickness=2)
    im_pose = cv2.resize(im_pose,(W,H))
    
    with torch.no_grad():
        c = model.get_learned_conditioning([prompt + ', ' + pos_prompt])
        nc = model.get_learned_conditioning([neg_prompt])
        # extract condition features
        pose = img2tensor(im_pose, bgr2rgb=True, float32=True)/255.
        pose = pose.unsqueeze(0)
        features_adapter = model_ad(pose.to(device))

        shape = [4, W//8, H//8]

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
        x_samples_ddim = x_samples_ddim.to('cpu')
        x_samples_ddim = x_samples_ddim.permute(0, 2, 3, 1).numpy()[0]
        x_samples_ddim = 255.*x_samples_ddim
        x_samples_ddim = x_samples_ddim.astype(np.uint8)

    return [im_pose[:,:,::-1].astype(np.uint8), x_samples_ddim]

DESCRIPTION = '''# T2I-Adapter (Keypose)
[Paper](https://arxiv.org/abs/2302.08453)               [GitHub](https://github.com/TencentARC/T2I-Adapter) 

This gradio demo is for keypose-guided generation. The current functions include:
- Keypose to Image Generation
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
            pos_prompt = gr.Textbox(label="Positive Prompt",
            value = 'crafted, elegant, meticulous, magnificent, maximum details, extremely hyper aesthetic, intricately detailed')
            with gr.Row():
                type_in = gr.inputs.Radio(['Keypose', 'Image'], type="value", default='Image', label='Input Types\n (You can input an image or a keypose map)')
                fix_sample = gr.inputs.Radio(['True', 'False'], type="value", default='False', label='Fix Sampling\n (Fix the random seed to produce a fixed output)')
            run_button = gr.Button(label="Run")
            con_strength = gr.Slider(label="Controling Strength (The guidance strength of the keypose to the result)", minimum=0, maximum=1, value=1, step=0.1)
            scale = gr.Slider(label="Guidance Scale (Classifier free guidance)", minimum=0.1, maximum=30.0, value=7.5, step=0.1)
            base_model = gr.inputs.Radio(['sd-v1-4.ckpt', 'anything-v4.0-pruned.ckpt'], type="value", default='sd-v1-4.ckpt', label='The base model you want to use')
        with gr.Column():
            result = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
        ips = [input_img, type_in, prompt, neg_prompt, pos_prompt, fix_sample, scale, con_strength, base_model]
    run_button.click(fn=process, inputs=ips, outputs=[result])

block.launch(server_name='0.0.0.0')
                
