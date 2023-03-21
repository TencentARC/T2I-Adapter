import torch
import numpy as np
from transformers import CLIPVisionModel

from ldm.models.diffusion.ddpm import LatentDiffusion, disabled_train
from ldm.util import instantiate_from_config


class T2IAdapterStyleV3(LatentDiffusion):

    def __init__(self, adapter_config, extra_cond_key, noise_schedule, *args, **kwargs):
        super(T2IAdapterStyleV3, self).__init__(*args, **kwargs)
        self.adapter = instantiate_from_config(adapter_config)
        self.extra_cond_key = extra_cond_key
        self.noise_schedule = noise_schedule
        self.clip_vision_model = CLIPVisionModel.from_pretrained(
            'openai/clip-vit-large-patch14'
        )

        self.clip_vision_model = self.clip_vision_model.eval()
        self.clip_vision_model.train = disabled_train
        for param in self.clip_vision_model.parameters():
            param.requires_grad = False


    def shared_step(self, batch, **kwargs):
        for k in self.ucg_training:
            if k == self.extra_cond_key:
                continue
            p = self.ucg_training[k]
            for i in range(len(batch[k])):
                if self.ucg_prng.choice(2, p=[1 - p, p]):
                    if isinstance(batch[k], list):
                        batch[k][i] = ""
        batch['jpg'] = batch['jpg'] * 2 - 1
        x, c = self.get_input(batch, self.first_stage_key)
        extra_cond = super(LatentDiffusion, self).get_input(batch, self.extra_cond_key).to(self.device)
        extra_cond = self.clip_vision_model(extra_cond)['last_hidden_state']
        features_adapter = self.adapter(extra_cond)
        if self.extra_cond_key in self.ucg_training:
            idx = np.random.choice(self.adapter.num_token, np.random.randint(1, self.adapter.num_token+1), replace=False)
            idx_tensor = torch.from_numpy(idx).to(features_adapter.device)
            features_adapter = torch.index_select(features_adapter, 1, idx_tensor)
        t = self.get_time_with_schedule(self.noise_schedule, x.size(0))
        loss, loss_dict = self(x, c, t=t, append_to_context=features_adapter)
        return loss, loss_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.adapter.parameters())
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
