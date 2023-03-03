import cv2
import torch
from enum import Enum, unique
from PIL import Image
from torch import autocast

from basicsr.utils import img2tensor
from ldm.util import resize_numpy_image


@unique
class ExtraCondition(Enum):
    sketch = 0
    keypose = 1
    seg = 2
    depth = 3
    canny = 4
    style = 5


def get_cond_model(opt, cond_type: ExtraCondition):
    if cond_type == ExtraCondition.sketch:
        from ldm.modules.extra_condition.model_edge import pidinet
        model = pidinet()
        ckp = torch.load('models/table5_pidinet.pth', map_location='cpu')['state_dict']
        model.load_state_dict({k.replace('module.', ''): v for k, v in ckp.items()}, strict=True)
        model.to(opt.device)
        return model
    elif cond_type == ExtraCondition.seg:
        raise NotImplementedError
    elif cond_type == ExtraCondition.keypose:
        import mmcv
        from mmdet.apis import init_detector
        from mmpose.apis import init_pose_model
        det_config = 'configs/mm/faster_rcnn_r50_fpn_coco.py'
        det_checkpoint = 'models/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
        pose_config = 'configs/mm/hrnet_w48_coco_256x192.py'
        pose_checkpoint = 'models/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
        det_config_mmcv = mmcv.Config.fromfile(det_config)
        det_model = init_detector(det_config_mmcv, det_checkpoint, device=opt.device)
        pose_config_mmcv = mmcv.Config.fromfile(pose_config)
        pose_model = init_pose_model(pose_config_mmcv, pose_checkpoint, device=opt.device)
        return {'pose_model': pose_model, 'det_model': det_model}
    elif cond_type == ExtraCondition.depth:
        from ldm.modules.extra_condition.midas.api import MiDaSInference
        model = MiDaSInference(model_type='dpt_hybrid').to(opt.device)
        return model
    elif cond_type == ExtraCondition.canny:
        return None
    elif cond_type == ExtraCondition.style:
        from transformers import CLIPProcessor, CLIPVisionModel
        version = '/group/30042/yanzewu/share_bwtween_machines/cache/models--openai--clip-vit-large-patch14/snapshots/8d052a0f05efbaefbc9e8786ba291cfdf93e5bff'
        processor = CLIPProcessor.from_pretrained(version)
        clip_vision_model = CLIPVisionModel.from_pretrained(version).to(opt.device)
        return {'processor': processor, 'clip_vision_model': clip_vision_model}
    else:
        raise NotImplementedError


def get_cond_sketch(opt, cond_image, cond_inp_type, cond_model=None):
    if isinstance(cond_image, str):
        edge = cv2.imread(cond_image)
    else:
        edge = cond_image
    edge = resize_numpy_image(edge, max_resolution=opt.max_resolution, resize_short_edge=opt.resize_short_edge)
    opt.H, opt.W = edge.shape[:2]
    if cond_inp_type == 'sketch':
        edge = img2tensor(edge)[0].unsqueeze(0).unsqueeze(0) / 255.
        edge = edge.to(opt.device)
    elif cond_inp_type == 'image':
        edge = img2tensor(edge).unsqueeze(0) / 255.
        edge = cond_model(edge.to(opt.device))[-1]
    else:
        raise NotImplementedError

    # edge = 1-edge # for white background
    edge = edge > 0.5
    edge = edge.float()

    return edge


def get_cond_seg(opt, cond_image, cond_inp_type='image', cond_model=None):
    if isinstance(cond_image, str):
        seg = cv2.imread(cond_image)
    else:
        seg = cond_image
    seg = resize_numpy_image(seg, max_resolution=opt.max_resolution, resize_short_edge=opt.resize_short_edge)
    opt.H, opt.W = seg.shape[:2]
    if cond_inp_type == 'seg':
        seg = img2tensor(seg).unsqueeze(0) / 255.
        seg = seg.to(opt.device)
    else:
        raise NotImplementedError

    return seg


def get_cond_keypose(opt, cond_image, cond_inp_type='image', cond_model=None):
    if isinstance(cond_image, str):
        pose = cv2.imread(cond_image)
    else:
        pose = cond_image
    pose = resize_numpy_image(pose, max_resolution=opt.max_resolution, resize_short_edge=opt.resize_short_edge)
    opt.H, opt.W = pose.shape[:2]
    if cond_inp_type == 'keypose':
        pose = img2tensor(pose).unsqueeze(0) / 255.
        pose = pose.to(opt.device)
    elif cond_inp_type == 'image':
        from mmdet.apis import inference_detector
        from mmpose.apis import inference_top_down_pose_model, process_mmdet_results
        from ldm.modules.extra_condition.utils import imshow_keypoints

        with autocast("cuda", dtype=torch.float32):
            mmdet_results = inference_detector(cond_model['det_model'], pose)
            # keep the person class bounding boxes.
            person_results = process_mmdet_results(mmdet_results, 1)

            # optional
            return_heatmap = False
            dataset = cond_model['pose_model'].cfg.data['test']['type']

            # e.g. use ('backbone', ) to return backbone feature
            output_layer_names = None
            pose_results, returned_outputs = inference_top_down_pose_model(
                cond_model['pose_model'],
                pose,
                person_results,
                bbox_thr=0.2,
                format='xyxy',
                dataset=dataset,
                dataset_info=None,
                return_heatmap=return_heatmap,
                outputs=output_layer_names)

        # show the results
        pose = imshow_keypoints(pose, pose_results, radius=2, thickness=2)
        pose = img2tensor(pose).unsqueeze(0) / 255.
        pose = pose.to(opt.device)
    else:
        raise NotImplementedError

    return pose


def get_cond_depth(opt, cond_image, cond_inp_type='image', cond_model=None):
    if isinstance(cond_image, str):
        depth = cv2.imread(cond_image)
    else:
        depth = cond_image
    depth = resize_numpy_image(depth, max_resolution=opt.max_resolution, resize_short_edge=opt.resize_short_edge)
    opt.H, opt.W = depth.shape[:2]
    if cond_inp_type == 'depth':
        depth = img2tensor(depth).unsqueeze(0) / 255.
        depth = depth.to(opt.device)
    elif cond_inp_type == 'image':
        depth = img2tensor(depth).unsqueeze(0) / 127.5 - 1.0
        depth = cond_model(depth.to(opt.device)).repeat(1, 3, 1, 1)
        depth -= torch.min(depth)
        depth /= torch.max(depth)
    else:
        raise NotImplementedError

    return depth


def get_cond_canny(opt, cond_image, cond_inp_type='image', cond_model=None):
    if isinstance(cond_image, str):
        canny = cv2.imread(cond_image)
    else:
        canny = cond_image
    canny = resize_numpy_image(canny, max_resolution=opt.max_resolution, resize_short_edge=opt.resize_short_edge)
    opt.H, opt.W = canny.shape[:2]
    if cond_inp_type == 'canny':
        canny = img2tensor(canny)[0:1].unsqueeze(0) / 255.
        canny = canny.to(opt.device)
    elif cond_inp_type == 'image':
        canny = cv2.Canny(canny, 100, 200)[..., None]
        canny = img2tensor(canny).unsqueeze(0) / 255.
        canny = canny.to(opt.device)
    else:
        raise NotImplementedError

    return canny


def get_cond_style(opt, cond_image, cond_inp_type='image', cond_model=None):
    assert cond_inp_type == 'image'
    if isinstance(cond_image, str):
        style = Image.open(cond_image)
    else:
        # cv2 image to PIL image
        style = Image.fromarray(cv2.cvtColor(cond_image, cv2.COLOR_BGR2RGB))

    style_for_clip = cond_model['processor'](images=style, return_tensors="pt")['pixel_values']
    style_feat = cond_model['clip_vision_model'](style_for_clip.to(opt.device))['last_hidden_state']

    return style_feat


def get_adapter_feature(inputs, adapters):
    ret = None
    if not isinstance(inputs, list):
        inputs = [inputs]
        adapters = [adapters]

    for input, adapter in zip(inputs, adapters):
        cur_feature_list = adapter['model'](input)
        if ret is None:
            ret = list(map(lambda x: x * adapter['cond_weight'], cur_feature_list))
        else:
            ret = list(map(lambda x, y: x + y * adapter['cond_weight'], ret, cur_feature_list))

    return ret
