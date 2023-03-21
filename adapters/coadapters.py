import torch
import torch.nn as nn
import numpy as np
from transformers import CLIPVisionModel

from ldm.models.diffusion.ddpm import LatentDiffusion, disabled_train
from ldm.util import instantiate_from_config
from ldm.modules.extra_condition.midas.api import MiDaSInference
from ldm.modules.extra_condition.model_edge import pidinet
from ldm.inference_base import read_state_dict


class CoAdapter(LatentDiffusion):

    def __init__(self, adapter_configs, coadapter_fuser_config, noise_schedule, *args, **kwargs):
        super(CoAdapter, self).__init__(*args, **kwargs)
        self.adapters = nn.ModuleDict()
        for adapter_config in adapter_configs:
            cond_name = adapter_config['cond_name']
            self.adapters[cond_name] = instantiate_from_config(adapter_config)
            if 'pretrained' in adapter_config:
                self.load_pretrained_adapter(cond_name, adapter_config['pretrained'])
        self.coadapter_fuser = instantiate_from_config(coadapter_fuser_config)
        self.training_adapters = list(self.adapters.keys())
        self.noise_schedule = noise_schedule

        # clip vision model as style model backbone
        self.clip_vision_model = CLIPVisionModel.from_pretrained(
            'openai/clip-vit-large-patch14'
        )
        self.clip_vision_model = self.clip_vision_model.eval()
        self.clip_vision_model.train = disabled_train
        for param in self.clip_vision_model.parameters():
            param.requires_grad = False

        # depth model
        self.midas_model = MiDaSInference(model_type='dpt_hybrid')
        self.midas_model = self.midas_model.eval()
        self.midas_model.train = disabled_train
        for param in self.midas_model.parameters():
            param.requires_grad = False

        # sketch model
        self.sketch_model = pidinet()
        ckp = torch.load('models/table5_pidinet.pth', map_location='cpu')['state_dict']
        self.sketch_model.load_state_dict({k.replace('module.', ''): v for k, v in ckp.items()}, strict=True)
        self.sketch_model = self.sketch_model.eval()
        self.sketch_model.train = disabled_train
        for param in self.sketch_model.parameters():
            param.requires_grad = False

    def load_pretrained_adapter(self, cond_name, pretrained_path):
        print(f'loading adapter {cond_name} from {pretrained_path}')
        state_dict = read_state_dict(pretrained_path)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('adapter.'):
                new_state_dict[k[len('adapter.'):]] = v
            else:
                new_state_dict[k] = v
        self.adapters[cond_name].load_state_dict(new_state_dict)

    @torch.inference_mode()
    def data_preparation_on_gpu(self, batch):
        # style
        style = batch['style'].to(self.device)
        style = self.clip_vision_model(style)['last_hidden_state']
        batch['style'] = style

        # depth
        depth = self.midas_model(batch['jpg']).repeat(1, 3, 1, 1)  # jpg range [-1, 1]
        for i in range(depth.size(0)):
            depth[i] -= torch.min(depth[i])
            depth[i] /= torch.max(depth[i])
        batch['depth'] = depth

        # sketch
        edge = 0.5 * batch['jpg'] + 0.5  # [-1, 1] to [0, 1]
        edge = self.sketch_model(edge)[-1]
        edge = edge > 0.5
        batch['sketch'] = edge.float()

    def get_adapter_features(self, batch, keep_conds):
        features = dict()

        for cond_name in keep_conds:
            if cond_name in batch:
                features[cond_name] = self.adapters[cond_name](batch[cond_name])

        return features

    def shared_step(self, batch, **kwargs):
        for k in self.ucg_training:
            p = self.ucg_training[k]
            for i in range(len(batch[k])):
                if self.ucg_prng.choice(2, p=[1 - p, p]):
                    if isinstance(batch[k], list):
                        batch[k][i] = ""
        batch['jpg'] = batch['jpg'] * 2 - 1
        x, c = self.get_input(batch, self.first_stage_key)
        self.data_preparation_on_gpu(batch)

        p = np.random.rand()
        if p < 0.1:
            keep_conds = self.training_adapters
        elif p < 0.2:
            keep_conds = []
        else:
            keep = np.random.choice(2, len(self.training_adapters), p=[0.5, 0.5])
            keep_conds = [cond_name for k, cond_name in zip(keep, self.training_adapters) if k == 1]

        features = self.get_adapter_features(batch, keep_conds)
        features_adapter, append_to_context = self.coadapter_fuser(features)

        t = self.get_time_with_schedule(self.noise_schedule, x.size(0))
        loss, loss_dict = self(x, c, t=t, features_adapter=features_adapter, append_to_context=append_to_context)
        return loss, loss_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.adapters.parameters()) + list(self.coadapter_fuser.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def on_save_checkpoint(self, checkpoint):
        keys = list(checkpoint['state_dict'].keys())
        for key in keys:
            if 'adapter' not in key:
                del checkpoint['state_dict'][key]

    def on_load_checkpoint(self, checkpoint):
        for name in self.state_dict():
            if 'adapter' not in name:
                checkpoint['state_dict'][name] = self.state_dict()[name]
