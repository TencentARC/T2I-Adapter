<p align="center">
  <img src="../assets/logo_coadapter.png" height=60>
</p>

## Overview

<p align="center">
  <img src="https://user-images.githubusercontent.com/17445847/225639246-26ee67a9-a9d9-47e4-b3bf-813d570e3d96.png" height=360>
</p>

We introduce **CoAdapter** (**Co**mposable **Adapter**) by jointly training T2I-Adapters and an extra fuser. The fuser allows different adapters with various conditions to be aware of each other and synergize to achieve more powerful composability, especially the combination of element-level style and other structural information.

CoAdapter is inspired by [Composer](https://github.com/damo-vilab/composer). However, instead of training the whole model, it only trains extra light-weight adapters based on T2I-Adapter. But CoAdapter can also show the capability of generating creative images with composibility. Note that the model is still in training and this release is only a preview.

## Demos

| Sketch                                                                                                                                    | Canny |                                                                   Depth                                                                   | Color (Spatial) | Style                                                                                                                                      | Results |
|:------------------------------------------------------------------------------------------------------------------------------------------|:-----:|:-----------------------------------------------------------------------------------------------------------------------------------------:|:---------------:|--------------------------------------------------------------------------------------------------------------------------------------------|---------|
| <img width="200" alt="image" src="https://user-images.githubusercontent.com/11482921/225659269-2d50e40d-f79b-41bc-9a0e-9dc73663f010.png"> |       |                                                                                                                                           |                 | <img width="100" alt="image" src="https://user-images.githubusercontent.com/11482921/225659792-07f3d5f4-3e26-4c52-988b-c4f228d6e45d.jpeg"> |    <img width="200" alt="image" src="https://user-images.githubusercontent.com/11482921/225660076-665b5889-3825-48cc-b9f9-06903fdd0c4b.jpg">     |
| <img width="200" alt="image" src="https://user-images.githubusercontent.com/11482921/225659269-2d50e40d-f79b-41bc-9a0e-9dc73663f010.png"> |       |                                                                                                                                           |                 | <img width="150" alt="image" src="https://user-images.githubusercontent.com/11482921/225660608-ca526f86-f506-4d0c-bdb8-75b95fbdbce0.png">  |    <img width="200" alt="image" src="https://user-images.githubusercontent.com/11482921/225660901-86bcbfd8-0643-4e17-a6ec-4a2305e5f0a5.png">     |
| <img width="200" alt="image" src="https://user-images.githubusercontent.com/11482921/225661058-656d87d7-3c8d-4216-820e-a02a8a5f5a4a.png"> |       | <img width="200" alt="image" src="https://user-images.githubusercontent.com/11482921/225661380-c9a01791-9b96-4b25-8878-87cdcf01a6f6.png"> |                 | <img width="250" alt="image" src="https://user-images.githubusercontent.com/11482921/225661180-98f338ee-950e-45d0-bd5f-4e8b7e82cecb.png">  |    <img width="200" alt="image" src="https://user-images.githubusercontent.com/11482921/225661273-82867799-b8f8-4fe5-9b12-cc99df85696d.png">     |
| <img width="200" alt="image" src="https://user-images.githubusercontent.com/11482921/225661058-656d87d7-3c8d-4216-820e-a02a8a5f5a4a.png"> |       | <img width="200" alt="image" src="https://user-images.githubusercontent.com/11482921/225661380-c9a01791-9b96-4b25-8878-87cdcf01a6f6.png"> |                 | <img width="250" alt="image" src="https://user-images.githubusercontent.com/11482921/225661755-cfa48f8d-712d-41a0-a1bf-4c1e90ca4d7a.png">  |    <img width="200" alt="image" src="https://user-images.githubusercontent.com/11482921/225661944-0f02ab3c-de8d-4898-a7b0-5ff55fa5f253.png">     |


## Benefits from CoAdapter

CoAdapter offers two advantages over the original T2I-Adapter. You can try CoAdapter in [![Huggingface CoAdapter](https://img.shields.io/static/v1?label=Demo&message=Huggingface%20Gradio&color=orange)](https://huggingface.co/spaces/Adapter/CoAdapter).

1. CoAdapter has improved **composability**, especially for the style modality, due to the joint training of multiple adapters.

| Input 1 | Input 2 |  Input3   | Prompt and seed | T2I-Adapter       | **CoAdapter** |
| :-----: | :-----: |  :-----:  |:-----: | :-----:    | :-----: |
|<img width="200" alt="image" src="https://user-images.githubusercontent.com/17445847/226340993-8858adb9-6161-4c5d-a579-1ac2ab88d31b.jpg"> <br>Canny: 1.0     |<img width="150" alt="image" src="https://user-images.githubusercontent.com/17445847/226341016-6d9990cf-8018-463d-80aa-7c29a52a0a87.jpg"> <br>Style: 1.0  |       |  "tower"<br>seed=42      |    <img width="250" alt="image" src="https://user-images.githubusercontent.com/17445847/226341027-5347281e-6a3e-4106-9f5d-f76809856952.png">         |  <img width="300" alt="image" src="https://user-images.githubusercontent.com/17445847/226341073-1b9ca9b1-ea23-4688-8553-416ebb7f09d1.png">       |
|<img width="200" alt="image" src="https://user-images.githubusercontent.com/17445847/226320934-bb332846-cccc-4aad-bf77-318a43eba6e2.jpg"> <br>Skecth: 1.0     |<img width="150" alt="image" src="https://user-images.githubusercontent.com/17445847/226320880-8ffe6e1d-de8a-45d2-9739-53024b86b1b2.png"> <br>Style: 1.0  |   <img width="150" alt="image" src="https://user-images.githubusercontent.com/17445847/226320922-588fa6c7-956d-4ce7-a707-a10f524d9c9c.png"> <br>Color: 1.0    |  "motorbike"<br>seed=993      |    <img width="250" alt="image" src="https://user-images.githubusercontent.com/17445847/226326697-12657ad0-dea8-4fd0-9410-70c8a1bc5304.png">         |  <img width="300" alt="image" src="https://user-images.githubusercontent.com/17445847/226321639-55cea2d5-aa6a-49ac-a647-c37d9b75009e.png">       |
|<img width="200" alt="image" src="https://user-images.githubusercontent.com/17445847/226330248-2a61d1c0-c39a-4d84-b3c0-61dc7613ae39.jpg"> <br>Skecth: 1.0     |<img width="150" alt="image" src="https://user-images.githubusercontent.com/17445847/226330258-d6fda31a-8631-4724-a342-93ea5269a74b.png"> <br>Style: 1.0  |       |  "a corgi"<br>seed=42      |    <img width="250" alt="image" src="https://user-images.githubusercontent.com/17445847/226330281-b536fc6a-486e-4c7a-b0bb-bffdb12e66c7.png">         |  <img width="300" alt="image" src="https://user-images.githubusercontent.com/17445847/226330294-025dc300-78d1-46f0-b7c6-3d03c2595e10.png">       |

2. The joint training of CoAdapter can also enhance the generation quality of each individual adapter.
   TODO

The above results are based on *coadapter-sd15v1* and *t2iadapter-sd14v1*.

## Useful Tips
- **Condition weight is important**.  If the generated image is not well aligned with the condition, increase the corresponding `Condition weight`. If increasing `Condition weight` is ineffective or degrades image quality, try decreasing the `Condition weight` of other conditions.
- **Start with fewer conditions**. If you plan to use more than two conditions to control the model, it is recommended to start with only one or two conditions. Once you have found the suitable `Condition weight` for existing conditions, gradually append the new conditions.
- It is recommended to use a step size of 0.1 to adjust `Condition weight`. From experience, `Condition weight` will not be less than 0.5 or greater than 1.5

## Training
```bash
python train.py -t --base configs/pl_train/coadapter-v1-train.yaml --gpus 0,1,2,3,4,5,6,7 --scale_lr False --num_nodes 1 --sd_finetune_from models/v1-5-pruned-emaonly.ckpt --name coadapter-v1-train --auto_resume
```
