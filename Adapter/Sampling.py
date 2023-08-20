from transformers import AutoTokenizer
from diffusers import DDPMScheduler, AutoencoderKL
import torch
from pytorch_lightning import seed_everything
import tqdm
import copy
import random
from basicsr.utils import tensor2img
import numpy as np

from Adapter.utils import import_model_class_from_model_name_or_path
from models.unet import UNet

class diffusion_inference:
    def __init__(self, model_id):
        self.device = 'cuda'
        self.model_id = model_id

        # load unet model
        self.scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.model = UNet.from_pretrained(model_id, subfolder="unet").to(self.device)
        try:
            self.model.enable_xformers_memory_efficient_attention()
        except:
            print('The current xformers is not compatible, please reinstall xformers to speed up.')
        self.scheduler.set_timesteps(50)

        tokenizer_one = AutoTokenizer.from_pretrained(
            self.model_id, subfolder="tokenizer", revision=None, use_fast=False
        )
        tokenizer_two = AutoTokenizer.from_pretrained(
            self.model_id, subfolder="tokenizer_2", revision=None, use_fast=False
        )

        # import correct text encoder classes
        text_encoder_cls_one = import_model_class_from_model_name_or_path(
            self.model_id, None
        )
        text_encoder_cls_two = import_model_class_from_model_name_or_path(
            self.model_id, None, subfolder="text_encoder_2"
        )

        # Load scheduler and models
        text_encoder_one = text_encoder_cls_one.from_pretrained(
            self.model_id, subfolder="text_encoder", revision=None
        )
        text_encoder_two = text_encoder_cls_two.from_pretrained(
            self.model_id, subfolder="text_encoder_2", revision=None
        )
        # self.text_encoders = [text_encoder_one.to(self.device), text_encoder_two.to(self.device)]
        self.text_encoders = [text_encoder_one, text_encoder_two]
        self.tokenizers = [tokenizer_one, tokenizer_two]
        self.vae = AutoencoderKL.from_pretrained(
                self.model_id,
                subfolder="vae",
                revision=None,
            )#.to(self.device)

    def reset_schedule(self, timesteps):
        self.scheduler.set_timesteps(timesteps)

    def inference(self, prompt, size, prompt_n='', adapter_features=None, guidance_scale=7.5, seed=-1, steps=50):
        prompt_batch = [prompt_n, prompt]
        prompt_embeds, unet_added_cond_kwargs = self.compute_embeddings(
            prompt_batch=prompt_batch,proportion_empty_prompts=0,text_encoders=self.text_encoders,tokenizers=self.tokenizers,size=size
        )
        self.reset_schedule(steps)
        if seed != -1:
            seed_everything(seed)
        noisy_latents = torch.randn((1, 4, size[0]//8, size[1]//8)).to("cuda")

        with torch.no_grad():
            for t in tqdm.tqdm(self.scheduler.timesteps):
                with torch.no_grad():
                    input = torch.cat([noisy_latents]*2)
                    noise_pred = self.model(
                            input,
                            t,
                            encoder_hidden_states=prompt_embeds["prompt_embeds"],
                            added_cond_kwargs=unet_added_cond_kwargs,
                            down_block_additional_residuals=copy.deepcopy(adapter_features),
                        )[0]
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    noisy_latents = self.scheduler.step(noise_pred, t, noisy_latents)[0]

        image = self.vae.decode(noisy_latents.cpu() / self.vae.config.scaling_factor, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = tensor2img(image)

        return image


    def encode_prompt(self, prompt_batch, proportion_empty_prompts, is_train=True):
        prompt_embeds_list = []

        captions = []
        for caption in prompt_batch:
            if random.random() < proportion_empty_prompts:
                captions.append("")
            elif isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])

        with torch.no_grad():
            for tokenizer, text_encoder in zip(self.tokenizers, self.text_encoders):
                text_inputs = tokenizer(
                    captions,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids
                prompt_embeds = text_encoder(
                    text_input_ids.to(text_encoder.device),
                    output_hidden_states=True,
                )

                # We are only ALWAYS interested in the pooled output of the final text encoder
                pooled_prompt_embeds = prompt_embeds[0]
                prompt_embeds = prompt_embeds.hidden_states[-2]
                bs_embed, seq_len, _ = prompt_embeds.shape
                prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
                prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
        return prompt_embeds, pooled_prompt_embeds

    def compute_embeddings(self, prompt_batch, proportion_empty_prompts, text_encoders, tokenizers, size, is_train=True):
        original_size = size
        target_size = size
        crops_coords_top_left = (0, 0)

        prompt_embeds, pooled_prompt_embeds = self.encode_prompt(
            prompt_batch, proportion_empty_prompts, is_train
        )
        add_text_embeds = pooled_prompt_embeds

        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])

        prompt_embeds = prompt_embeds.to(self.device)
        add_text_embeds = add_text_embeds.to(self.device)
        add_time_ids = add_time_ids.repeat(len(prompt_batch), 1)
        add_time_ids = add_time_ids.to(self.device, dtype=prompt_embeds.dtype)
        unet_added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

        return {"prompt_embeds": prompt_embeds}, unet_added_cond_kwargs

    
