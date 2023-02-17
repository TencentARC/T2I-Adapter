# <div align="center"><b>T2I-Adapter</a></b></div>

<div align="center">

⏬[**Download Models**](#-download-models) **|** 💻[**How to Test**](#-how-to-test)

</div>

Official implementation of T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models.

#### [Paper](https://arxiv.org/abs/2302.08453)

<p align="center">
  <img src="assets/overview1.png" height=250>
</p>

We propose T2I-Adapter, a **simple and small (~70M parameters, ~300M storage space)** network that can provide extra guidance to pre-trained text-to-image models while **freezing** the original large text-to-image models.

T2I-Adapter aligns internal knowledge in T2I models with external control signals.
We can train various adapters according to different conditions, and achieve rich control and editing effects.

<p align="center">
  <img src="assets/teaser.png" height=500>
</p>

### ⏬ Download Models

Put the downloaded models in the `T2I-Adapter/models` folder.

1. The **T2I-Adapters** can be download from <https://huggingface.co/TencentARC/T2I-Adapter>.
2. The pretrained **Stable Diffusion v1.4** models can be download from <https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/tree/main>. You need to download the `sd-v1-4.ckpt
` file.
3. [Optional] If you want to use **Anything v4.0** models, you can download the pretrained models from <https://huggingface.co/andite/anything-v4.0/tree/main>. You need to download the `anything-v4.0-pruned.ckpt` file.
4. The pretrained **clip-vit-large-patch14** folder can be download from <https://huggingface.co/openai/clip-vit-large-patch14/tree/main>. Remember to download the whole folder!

After downloading, the folder structure should be like this:

<p align="center">
  <img src="assets/downloaded_models.png" height=100>
</p>

### 🔧 Dependencies and Installation

- Python >= 3.6 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.4](https://pytorch.org/)
```bash
pip install -r requirements.txt
```

### 💻 How to Test

- The results are in the `experiments` folder.
- If you want to use the `Anything v4.0`, please add `--ckpt models/anything-v4.0-pruned.ckpt` in the following commands.

#### Sketch Adapter

- Sketch to Image Transition

> python -m torch.distributed.launch --nproc_per_node=1 test_sketch.py --plms --auto_resume --prompt "A car with flying wings" --path_cond examples/sketch/car.png --ckpt models/sd-v1-4.ckpt --type_in sketch

- Image to Image Transition

> python -m torch.distributed.launch --nproc_per_node=1 test_sketch.py --plms --auto_resume --prompt "A beautiful girl" --path_cond examples/anything_sketch/human.png --ckpt models/sd-v1-4.ckpt --type_in image

- Image to Image Transition with **Anything** setting

> python -m torch.distributed.launch --nproc_per_node=1 test_sketch.py --plms --auto_resume --prompt "A beautiful girl" --path_cond examples/anything_sketch/human.png --ckpt models/anything-v4.0-pruned.ckpt --type_in image

#### Keypose Adapter

> python -m torch.distributed.launch --nproc_per_node=1 test_keypose.py --plms --auto_resume --prompt "An Iron man" --path_cond examples/keypose/iron.png

#### Segmentation Adapter

> python -m torch.distributed.launch --nproc_per_node=1 test_seg.py --plms --auto_resume --prompt "A black Honda motorcycle parked in front of a garage" --path_cond examples/seg/motor.png

#### Two adapters: Segmentation and Sketch Adapters

> python -m torch.distributed.launch --nproc_per_node=1 test_seg_sketch.py --plms --auto_resume --prompt "An all white kitchen with an electric stovetop" --path_cond examples/seg_sketch/mask.png --path_cond2 examples/seg_sketch/edge.png

#### Local editing with adapters
>
> python -m torch.distributed.launch --nproc_per_node=1 test_sketch_edit.py --plms --auto_resume --prompt "A white cat" --path_cond examples/edit_cat/edge_2.png --path_x0 examples/edit_cat/im.png --path_mask examples/edit_cat/mask.png

## Stable Diffusion + T2I-Adapters (only ~70M parameters, ~300M storage space)

The following is the detailed structure of a **Stable Diffusion** model with the **T2I-Adapter**.
<p align="center">
  <img src="assets/overview2.png" height=300>
</p>

<!-- ## Web Demo

* All the usage of three T2I-Adapters (i.e, sketch, keypose and segmentation) are integrated into [Huggingface Spaces]() 🤗 using [Gradio](). Have fun with the Web Demo.  -->

## 🚀 Interesting Applications

### Stable Diffusion results guided with the sketch T2I-Adapter

The corresponding edge maps are predicted by PiDiNet. The sketch T2I-Adapter can well generalize to other similar sketch types, for example, sketches from the Internet and user scribbles.

<p align="center">
  <img src="assets/sketch_base.png" height=800>
</p>

### Stable Diffusion results guided with the keypose T2I-Adapter

The keypose results predicted by the [MMPose](https://github.com/open-mmlab/mmpose).
With the keypose guidance, the keypose T2I-Adapter can also help to generate animals with the same keypose, for example, pandas and tigers.

<p align="center">
  <img src="assets/keypose_base.png" height=600>
</p>

### T2I-Adapter with Anything-v4.0

Once the T2I-Adapter is trained, it can act as a **plug-and-play module** and can be seamlessly integrated into the finetuned diffusion models **without re-training**, for example, Anything-4.0.

#### ✨ Anything results with the plug-and-play sketch T2I-Adapter (no extra training)

<p align="center">
  <img src="assets/sketch_anything.png" height=600>
</p>

#### Anything results with the plug-and-play keypose T2I-Adapter (no extra training)

<p align="center">
  <img src="assets/keypose_anything.png" height=600>
</p>

### Local editing with the sketch adapter

When combined with the inpaiting mode of Stable Diffusion, we can realize local editing with user specific guidance.

#### ✨ Change the head direction of the cat

<p align="center">
  <img src="assets/local_editing_cat.png" height=300>
</p>

#### ✨ Add rabbit ears on the head of the Iron Man.

<p align="center">
  <img src="assets/local_editing_ironman.png" height=400>
</p>

### Combine different concepts with adapter

Adapter can be used to enhance the SD ability to combine different concepts.

####  ✨ A car with flying wings. / A doll in the shape of letter ‘A’.

<p align="center">
  <img src="assets/enhance_SD2.png" height=600>
</p>

### Sequential editing with the sketch adapter

We can realize the sequential editing with the adapter guidance.

<p align="center">
  <img src="assets/sequential_edit.png">
</p>

### Composable Guidance with multiple adapters

Stable Diffusion results guided with the segmentation and sketch adapters together.

<p align="center">
  <img src="assets/multiple_adapters.png">
</p>


![visitors](https://visitor-badge.glitch.me/badge?page_id=TencentARC/T2I-Adapter)
