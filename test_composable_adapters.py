import cv2
import os
import torch
from pytorch_lightning import seed_everything
from torch import autocast

from basicsr.utils import tensor2img
from ldm.inference_base import diffusion_inference, get_adapters, get_base_argument_parser, get_sd_models
from ldm.modules.extra_condition import api
from ldm.modules.extra_condition.api import ExtraCondition, get_adapter_feature, get_cond_model

torch.set_grad_enabled(False)


def main():
    supported_cond = [e.name for e in ExtraCondition]
    parser = get_base_argument_parser()
    for cond_name in supported_cond:
        parser.add_argument(
            f'--{cond_name}_path',
            type=str,
            default=None,
            help=f'condition image path for {cond_name}',
        )
        parser.add_argument(
            f'--{cond_name}_inp_type',
            type=str,
            default='image',
            help=f'the type of the input condition image, can be image or {cond_name}',
            choices=['image', cond_name],
        )
        parser.add_argument(
            f'--{cond_name}_adapter_ckpt',
            type=str,
            default=None,
            help=f'path to checkpoint of the {cond_name} adapter, '
                 f'if {cond_name}_path is not None, this should not be None too',
        )
        parser.add_argument(
            f'--{cond_name}_weight',
            type=float,
            default=1.0,
            help=f'the {cond_name} adapter features are multiplied by the {cond_name}_weight and then summed up together',
        )
    opt = parser.parse_args()

    # process argument
    activated_conds = []
    cond_paths = []
    adapter_ckpts = []
    for cond_name in supported_cond:
        if getattr(opt, f'{cond_name}_path') is None:
            continue
        assert getattr(opt, f'{cond_name}_adapter_ckpt') is not None, f'you should specify the {cond_name}_adapter_ckpt'
        activated_conds.append(cond_name)
        cond_paths.append(getattr(opt, f'{cond_name}_path'))
        adapter_ckpts.append(getattr(opt, f'{cond_name}_adapter_ckpt'))
    assert len(activated_conds) != 0, 'you did not input any condition'

    if opt.outdir is None:
        opt.outdir = f'outputs/test-composable-adapters'
    os.makedirs(opt.outdir, exist_ok=True)
    if opt.resize_short_edge is None:
        print(f"you don't specify the resize_shot_edge, so the maximum resolution is set to {opt.max_resolution}")
    opt.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # prepare models
    adapters = []
    cond_models = []
    cond_inp_types = []
    process_cond_modules = []
    for cond_name in activated_conds:
        adapters.append(get_adapters(opt, getattr(ExtraCondition, cond_name)))
        cond_inp_type = getattr(opt, f'{cond_name}_inp_type', 'image')
        if cond_inp_type == 'image':
            cond_models.append(get_cond_model(opt, getattr(ExtraCondition, cond_name)))
        else:
            cond_models.append(None)
        cond_inp_types.append(cond_inp_type)
        process_cond_modules.append(getattr(api, f'get_cond_{cond_name}'))
    sd_model, sampler = get_sd_models(opt)

    # inference
    with torch.inference_mode(), \
            sd_model.ema_scope(), \
            autocast('cuda'):
        seed_everything(opt.seed)
        conds = []
        for cond_idx, cond_name in enumerate(activated_conds):
            conds.append(process_cond_modules[cond_idx](
                opt, cond_paths[cond_idx], cond_inp_types[cond_idx], cond_models[cond_idx],
            ))
        adapter_features, append_to_context = get_adapter_feature(conds, adapters)
        for v_idx in range(opt.n_samples):
            result = diffusion_inference(opt, sd_model, sampler, adapter_features, append_to_context)
            base_count = len(os.listdir(opt.outdir))
            cv2.imwrite(os.path.join(opt.outdir, f'{base_count:05}_result.png'), tensor2img(result))


if __name__ == '__main__':
    main()
