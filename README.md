<p align="center">
  <img src="assets/logo3.png" height=65>
</p>


Official implementation of **[T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.08453)** based on [Stable Diffusion-XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0).


🚩 Due to the limited computing resources, those adapters are not fully trained. We are collaborating with [HuggingFace](https://huggingface.co/), and a more powerful adapter is in the works.

---

# 🔥🔥🔥 Why T2I-Adapter-XL? 
## The Original Recipe Drives Larger SD.

|   | SD-V1.4/1.5 | SD-XL | T2I-Adapter | T2I-Adapter-XL |
| --- | --- |--- |--- |--- |
| Parameters | 860M | 2.6B |77 M | 77 MB | |

## Inherit High-quality Generation from SDXL.

- Keypoint-guided
<p align="center">
  <img src="https://huggingface.co/TencentARC/T2I-Adapter/resolve/main/assets_XL/g_pose.png" height=520>
</p>

- Sketch-guided
<p align="center">
  <img src="https://huggingface.co/TencentARC/T2I-Adapter/resolve/main/assets_XL/g_sketch.PNG" height=520>
</p>

- Canny-guided
<p align="center">
  <img src="https://huggingface.co/TencentARC/T2I-Adapter/resolve/main/assets_XL/g_canny.png" height=520>
</p>

# 🔧 Dependencies and Installation

- Python >= 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 2.0.1](https://pytorch.org/)
```bash
pip install -r requirements.txt
```

# ⏬ Download Models 
All models will be automatically downloaded to the `checkpoints` folder. You can also choose to download manually from this [url](https://huggingface.co/TencentARC/T2I-Adapter/tree/main/models_XL).

# 🔥 How to Train
Here we take sketch guidance as an example, but of course, you can also prepare your own dataset following this method.
```bash
accelerate launch train_sketch.py --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 --output_dir experiments/adapter_sketch_xl --config configs/train/Adapter-XL-sketch.yaml --mixed_precision="fp16" --resolution=1024 --learning_rate=1e-5 --max_train_steps=60000 --train_batch_size=1 --gradient_accumulation_steps=4 --report_to="wandb" --seed=42 --num_train_epochs 100
```

We train with `FP16` data precision on `4` NVIDIA `A100` GPUs.

# 💻 How to Test
Inference requires at least `15GB` of GPU memory.

- Sketch example:

```bash
python test.py --prompt 'a photo of a dog in real world, high quality' --config configs/inference/Adapter-XL-sketch.yaml --path_source examples/dog.png --in_type image
```

- Canny example:

```bash
python test.py --prompt 'a photo of a dog in real world, high quality' --config configs/inference/Adapter-XL-canny.yaml --path_source examples/dog.png --in_type image
```

- Keypoint example:

```bash
python test.py --prompt 'a photo of two people in real world, high quality, clear' --config configs/inference/Adapter-XL-openpose.yaml --path_source examples/people.jpg --in_type image
```


Or you can directly use our Gradio Demo for more detailed generation design.

```bash
python app.py
```

# 🤗 Acknowledgements
- Thanks to HuggingFace for their support of T2I-Adapter.

- T2I-Adapter is co-hosted by Tencent ARC Lab and Peking University [VILLA](https://villa.jianzhang.tech/).





