import os
from tqdm import tqdm

import torch.hub

example_list = [
    'examples/canny/rabbit.png',
    'examples/canny/toy_canny.png',
    'examples/color/color_0000.png',
    'examples/color/color_0001.png',
    'examples/color/color_0002.png',
    'examples/color/color_0003.png',
    'examples/color/color_0004.png',
    'examples/depth/desk_depth.png',
    'examples/depth/sd.png',
    'examples/edit_cat/edge.png',
    'examples/edit_cat/edge_2.png',
    'examples/edit_cat/im.png',
    'examples/edit_cat/mask.png',
    'examples/keypose/iron.png',
    'examples/keypose/person_keypose.png',
    'examples/openpose/iron_man_image.png',
    'examples/openpose/iron_man_pose.png',
    'examples/seg/dinner.png',
    'examples/seg/motor.png',
    'examples/seg_sketch/edge.png',
    'examples/seg_sketch/mask.png',
    'examples/sketch/car.png',
    'examples/sketch/girl.jpeg',
    'examples/sketch/human.png',
    'examples/sketch/scenery.jpg',
    'examples/sketch/scenery2.jpg',
    'examples/style/Bianjing_city_gate.jpeg',
    'examples/style/Claude_Monet,_Impression,_soleil_levant,_1872.jpeg',
    'examples/style/DVwG-hevauxk1601457.jpeg',
    'examples/style/The_Persistence_of_Memory.jpeg',
    'examples/style/Tsunami_by_hokusai_19th_century.jpeg',
    'examples/style/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpeg',
    'examples/style/cyberpunk.png',
    'examples/style/scream.jpeg'
]

huggingface_root = 'https://huggingface.co/TencentARC/T2I-Adapter/resolve/main'

for example_path in tqdm(example_list):
    if not os.path.exists(example_path):
        os.makedirs(os.path.dirname(example_path), exist_ok=True)
        torch.hub.download_url_to_file(f'{huggingface_root}/{example_path}', example_path)
