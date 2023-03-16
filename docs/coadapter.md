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

1.

2.

## Training
