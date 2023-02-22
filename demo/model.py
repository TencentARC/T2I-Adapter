import torch
from basicsr.utils import img2tensor, tensor2img
from pytorch_lightning import seed_everything
from ldm.models.diffusion.plms import PLMSSampler
from ldm.modules.encoders.adapter import Adapter
from ldm.util import instantiate_from_config
from ldm.modules.structure_condition.model_edge import pidinet
import gradio as gr
from omegaconf import OmegaConf
import mmcv
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import (inference_top_down_pose_model, init_pose_model, process_mmdet_results, vis_pose_result)
import os
import cv2
import numpy as np


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
    _, _ = model.load_state_dict(sd, strict=False)

    model.cuda()
    model.eval()
    return model

class Model_all:
    def __init__(self, device='cpu'):
        # common part
        self.device = device
        self.config = OmegaConf.load("configs/stable-diffusion/app.yaml")
        self.config.model.params.cond_stage_config.params.device = device
        self.base_model = load_model_from_config(self.config, "models/sd-v1-4.ckpt").to(device)
        self.current_base_pose = 'sd-v1-4.ckpt'
        self.current_base_sketch = 'sd-v1-4.ckpt'
        self.sampler = PLMSSampler(self.base_model)

        # sketch part
        self.model_sketch = Adapter(channels=[320, 640, 1280, 1280][:4], nums_rb=2, ksize=1, sk=True, use_conv=False).to(device)
        self.model_sketch.load_state_dict(torch.load("models/t2iadapter_sketch_sd14v1.pth", map_location=device))
        self.model_edge = pidinet()
        ckp = torch.load('models/table5_pidinet.pth', map_location='cpu')['state_dict']
        self.model_edge.load_state_dict({k.replace('module.',''):v for k, v in ckp.items()})
        self.model_edge.to(device)

        # keypose part
        self.model_pose = Adapter(cin=int(3*64), channels=[320, 640, 1280, 1280][:4], nums_rb=2, ksize=1, sk=True, use_conv=False).to(device)
        self.model_pose.load_state_dict(torch.load("models/t2iadapter_keypose_sd14v1.pth", map_location=device))
        ## mmpose
        det_config = 'configs/mm/faster_rcnn_r50_fpn_coco.py' 
        det_checkpoint = 'models/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
        pose_config = 'configs/mm/hrnet_w48_coco_256x192.py'
        pose_checkpoint = 'models/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
        self.det_cat_id = 1
        self.bbox_thr = 0.2
        ## detector
        det_config_mmcv = mmcv.Config.fromfile(det_config)
        self.det_model = init_detector(det_config_mmcv, det_checkpoint, device=device)
        pose_config_mmcv = mmcv.Config.fromfile(pose_config)
        self.pose_model = init_pose_model(pose_config_mmcv, pose_checkpoint, device=device)
        ## color
        self.skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10],
            [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]
        self.pose_kpt_color = [[51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255], [0, 255, 0],
                        [255, 128, 0], [0, 255, 0], [255, 128, 0], [0, 255, 0], [255, 128, 0], [0, 255, 0], [255, 128, 0],
                        [0, 255, 0], [255, 128, 0], [0, 255, 0], [255, 128, 0]]
        self.pose_link_color = [[0, 255, 0], [0, 255, 0], [255, 128, 0], [255, 128, 0],
                        [51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255], [0, 255, 0], [255, 128, 0],
                        [0, 255, 0], [255, 128, 0], [51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255],
                        [51, 153, 255], [51, 153, 255], [51, 153, 255]]

    def load_vae(self):
        vae_sd = torch.load(os.path.join('models', 'anything-v4.0.vae.pt'), map_location="cpu")
        sd = vae_sd["state_dict"]
        self.base_model.first_stage_model.load_state_dict(sd, strict=False)

    def process_input(self, input_img):
        h, w = input_img.shape[:2]
        if w > h:
            W = int(w * 512 / h)
            H = 512
        else:
            H = int(h * 512 / w)
            W = 512
        W, H = map(lambda x: x - x % 64, (W, H))
        input_img = cv2.resize(input_img, (W, H))
        return input_img

    @torch.no_grad()
    def process_sketch(self, input_img, type_in, color_back, prompt, neg_prompt, pos_prompt, fix_sample, scale, cond_strength, base_model):
        if self.current_base_sketch != base_model:
            ckpt = os.path.join("models", base_model)
            pl_sd = torch.load(ckpt, map_location="cpu")
            if "state_dict" in pl_sd:
                sd = pl_sd["state_dict"]
            else:
                sd = pl_sd
            self.base_model.load_state_dict(sd, strict=False)
            if 'anything' in base_model.lower():
                self.load_vae()
            self.current_base_sketch = base_model
            # del sd
            # del pl_sd
        cond_strength = int((1-cond_strength)*50)
        if fix_sample == 'True':
            seed_everything(42)
        im = self.process_input(input_img)
        h, w = im.shape[:2]

        if type_in == 'Sketch':
            if color_back == 'White':
                im = 255-im
            im_edge = im.copy()
            im = img2tensor(im)[0].unsqueeze(0).unsqueeze(0)/255.
            im = im>0.5
            im = im.float()
        elif type_in == 'Image':
            im = img2tensor(im).unsqueeze(0)/255.
            im = self.model_edge(im.to(self.device))[-1]
            im = im>0.5
            im = im.float()
            im_edge = tensor2img(im)
        
        # save gpu memory
        self.base_model.model = self.base_model.model.cpu()
        self.model_sketch = self.model_sketch.cuda()
        self.base_model.first_stage_model = self.base_model.first_stage_model.cpu()
        self.base_model.cond_stage_model = self.base_model.cond_stage_model.cuda()

        # extract condition features
        c = self.base_model.get_learned_conditioning([prompt+', '+pos_prompt])
        nc = self.base_model.get_learned_conditioning([neg_prompt])
        features_adapter = self.model_sketch(im.to(self.device))
        shape = [4, h//8, w//8]
        
        # save gpu memory
        self.model_sketch = self.model_sketch.cpu()
        self.base_model.cond_stage_model = self.base_model.cond_stage_model.cpu()
        self.base_model.model = self.base_model.model.cuda()

        # sampling
        samples_ddim, _ = self.sampler.sample(S=50,
                                        conditioning=c,
                                        batch_size=1,
                                        shape=shape,
                                        verbose=False,
                                        unconditional_guidance_scale=scale,
                                        unconditional_conditioning=nc,
                                        eta=0.0,
                                        x_T=None,
                                        features_adapter=features_adapter,
                                        mode = 'sketch',
                                        cond_strength = cond_strength)
        # save gpu memory
        self.base_model.first_stage_model = self.base_model.first_stage_model.cuda()

        x_samples_ddim = self.base_model.decode_first_stage(samples_ddim)
        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        x_samples_ddim = x_samples_ddim.to('cpu')
        x_samples_ddim = x_samples_ddim.permute(0, 2, 3, 1).numpy()[0]
        x_samples_ddim = 255.*x_samples_ddim
        x_samples_ddim = x_samples_ddim.astype(np.uint8)

        return [im_edge, x_samples_ddim]

    @torch.no_grad()
    def process_draw(self, input_img, prompt, neg_prompt, pos_prompt, fix_sample, scale, cond_strength, base_model):
        if self.current_base_sketch != base_model:
            ckpt = os.path.join("models", base_model)
            pl_sd = torch.load(ckpt, map_location="cpu")
            if "state_dict" in pl_sd:
                sd = pl_sd["state_dict"]
            else:
                sd = pl_sd
            self.base_model.load_state_dict(sd, strict=False) #load_model_from_config(config, os.path.join("models", base_model)).to(device)
            if 'anything' in base_model.lower():
                self.load_vae()
            self.current_base_sketch = base_model
        cond_strength = int((1-cond_strength)*50)
        if fix_sample == 'True':
            seed_everything(42)
        input_img = input_img['mask']
        c = input_img[:, :, 0:3].astype(np.float32)
        a = input_img[:, :, 3:4].astype(np.float32) / 255.0
        im = c * a + 255.0 * (1.0 - a)
        im = im.clip(0, 255).astype(np.uint8)
        im = cv2.resize(im,(512,512))

        # im = 255-im
        im_edge = im.copy()
        im = img2tensor(im)[0].unsqueeze(0).unsqueeze(0)/255.
        im = im>0.5
        im = im.float()

        # save gpu memory
        self.base_model.model = self.base_model.model.cpu()
        self.model_sketch = self.model_sketch.cuda()
        self.base_model.first_stage_model = self.base_model.first_stage_model.cpu()
        self.base_model.cond_stage_model = self.base_model.cond_stage_model.cuda()
        
        # extract condition features
        c = self.base_model.get_learned_conditioning([prompt+', '+pos_prompt])
        nc = self.base_model.get_learned_conditioning([neg_prompt])
        features_adapter = self.model_sketch(im.to(self.device))
        shape = [4, 64, 64]

        # save gpu memory
        self.model_sketch = self.model_sketch.cpu()
        self.base_model.cond_stage_model = self.base_model.cond_stage_model.cpu()
        self.base_model.model = self.base_model.model.cuda()

        # sampling
        samples_ddim, _ = self.sampler.sample(S=50,
                                        conditioning=c,
                                        batch_size=1,
                                        shape=shape,
                                        verbose=False,
                                        unconditional_guidance_scale=scale,
                                        unconditional_conditioning=nc,
                                        eta=0.0,
                                        x_T=None,
                                        features_adapter=features_adapter,
                                        mode = 'sketch',
                                        cond_strength = cond_strength)
        
        # save gpu memory
        self.base_model.first_stage_model = self.base_model.first_stage_model.cuda()

        x_samples_ddim = self.base_model.decode_first_stage(samples_ddim)
        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        x_samples_ddim = x_samples_ddim.to('cpu')
        x_samples_ddim = x_samples_ddim.permute(0, 2, 3, 1).numpy()[0]
        x_samples_ddim = 255.*x_samples_ddim
        x_samples_ddim = x_samples_ddim.astype(np.uint8)

        return [im_edge, x_samples_ddim]

    @torch.no_grad()
    def process_keypose(self, input_img, type_in, prompt, neg_prompt, pos_prompt, fix_sample, scale, cond_strength, base_model):
        if self.current_base_pose != base_model:
            ckpt = os.path.join("models", base_model)
            pl_sd = torch.load(ckpt, map_location="cpu")
            if "state_dict" in pl_sd:
                sd = pl_sd["state_dict"]
            else:
                sd = pl_sd
            self.base_model.load_state_dict(sd, strict=False)
            if 'anything' in base_model.lower():
                self.load_vae()
            self.current_base_pose = base_model
        cond_strength = int((1-cond_strength)*50)
        if fix_sample == 'True':
            seed_everything(42)
        im = self.process_input(input_img)
        h, w = im.shape[:2]

        if type_in == 'Keypose':
            im_pose = im.copy()
            im = img2tensor(im).unsqueeze(0)/255.
        elif type_in == 'Image':
            image = im.copy()
            im = img2tensor(im).unsqueeze(0)/255.
            mmdet_results = inference_detector(self.det_model, image)
            # keep the person class bounding boxes.
            person_results = process_mmdet_results(mmdet_results, self.det_cat_id)

            # optional
            return_heatmap = False
            dataset = self.pose_model.cfg.data['test']['type']

            # e.g. use ('backbone', ) to return backbone feature
            output_layer_names = None
            pose_results, _ = inference_top_down_pose_model(
                self.pose_model,
                image,
                person_results,
                bbox_thr=self.bbox_thr,
                format='xyxy',
                dataset=dataset,
                dataset_info=None,
                return_heatmap=return_heatmap,
                outputs=output_layer_names)

            # show the results
            im_pose = imshow_keypoints(
                image,
                pose_results,
                skeleton=self.skeleton,
                pose_kpt_color=self.pose_kpt_color,
                pose_link_color=self.pose_link_color,
                radius=2,
                thickness=2)
        im_pose = cv2.resize(im_pose,(w,h))
        
        # save gpu memory
        self.base_model.model = self.base_model.model.cpu()
        self.model_pose = self.model_pose.cuda()
        self.base_model.first_stage_model = self.base_model.first_stage_model.cpu()
        self.base_model.cond_stage_model = self.base_model.cond_stage_model.cuda()

        # extract condition features
        c = self.base_model.get_learned_conditioning([prompt+', '+pos_prompt])
        nc = self.base_model.get_learned_conditioning([neg_prompt])
        pose = img2tensor(im_pose, bgr2rgb=True, float32=True)/255.
        pose = pose.unsqueeze(0)
        features_adapter = self.model_pose(pose.to(self.device))

        # save gpu memory
        self.model_pose = self.model_pose.cpu()
        self.base_model.cond_stage_model = self.base_model.cond_stage_model.cpu()
        self.base_model.model = self.base_model.model.cuda()

        shape = [4, h//8, w//8]

        # sampling
        samples_ddim, _ = self.sampler.sample(S=50,
                                        conditioning=c,
                                        batch_size=1,
                                        shape=shape,
                                        verbose=False,
                                        unconditional_guidance_scale=scale,
                                        unconditional_conditioning=nc,
                                        eta=0.0,
                                        x_T=None,
                                        features_adapter=features_adapter,
                                        mode = 'sketch',
                                        cond_strength = cond_strength)

        # save gpu memory
        self.base_model.first_stage_model = self.base_model.first_stage_model.cuda()

        x_samples_ddim = self.base_model.decode_first_stage(samples_ddim)
        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        x_samples_ddim = x_samples_ddim.to('cpu')
        x_samples_ddim = x_samples_ddim.permute(0, 2, 3, 1).numpy()[0]
        x_samples_ddim = 255.*x_samples_ddim
        x_samples_ddim = x_samples_ddim.astype(np.uint8)

        return [im_pose[:,:,::-1].astype(np.uint8), x_samples_ddim]

if __name__ == '__main__':
    model = Model_all('cpu')