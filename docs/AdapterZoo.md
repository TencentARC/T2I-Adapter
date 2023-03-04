# Adapter Zoo

You can download the adapters from <https://huggingface.co/TencentARC/T2I-Adapter/tree/main>

All the following adapters are trained with Stable Diffusion (SD) V1.4, and they can be directly used on custom models as long as they are fine-tuned from the same text-to-image models, such as Anything-4.0 or models on the <https://civitai.com/>.

| Adapter Name  | Adapter Description | Demos|Model Parameters|  Model Storage | |
| --- | --- |--- |--- |--- |---|
| t2iadapter_color_sd14v1.pth | Spatial color palette → image | [Demos](examples.md#color-adapter-spatial-palette) |18 M | 75 MB | |
| t2iadapter_style_sd14v1.pth | Image style → image | [Demos](examples.md#style-adapter)|| 154MB |  Preliminary model. Style adapters with finer controls are on the way|
| t2iadapter_openpose_sd14v1.pth | Openpose → image| [Demos](examples.md#openpose-adapter) |77 M| 309 MB | |
| t2iadapter_canny_sd14v1.pth | Canny edges → image | [Demos](examples.md#canny-adapter-edge )|77 M | 309 MB ||
| t2iadapter_sketch_sd14v1.pth | sketch → image ||77 M| 308 MB | |
| t2iadapter_keypose_sd14v1.pth | keypose → image || 77 M| 309 MB | mmpose style |
| t2iadapter_seg_sd14v1.pth | segmentation → image ||77 M| 309 MB ||
| t2iadapter_depth_sd14v1.pth | depth maps → image ||77 M | 309 MB | Not the final model, still under training|
