import torch
import json
import cv2
import torch
import os
from basicsr.utils import img2tensor, tensor2img
import random

class dataset_coco():
    def __init__(self, path_json, root_path, image_size, mode='train'):
        super(dataset_coco, self).__init__()
        with open(path_json, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        data = data['images']
        self.paths = []
        self.root_path = root_path
        for file in data:
            input_path = file['filepath']
            if mode == 'train':
                if 'val' not in input_path:
                    self.paths.append(file)
            else:
                if 'val' in input_path:
                    self.paths.append(file)

    def __getitem__(self, idx):
        file = self.paths[idx]
        input_path = file['filepath']
        input_name = file['filename']
        path = os.path.join(self.root_path, input_path, input_name)
        im = cv2.imread(path)
        im = cv2.resize(im, (512,512))
        im = img2tensor(im, bgr2rgb=True, float32=True)/255.
        sentences = file['sentences']
        sentence =  sentences[int(random.random()*len(sentences))]['raw'].strip('.')
        return {'im':im, 'sentence':sentence}

    def __len__(self):
        return len(self.paths)


class dataset_coco_mask():
    def __init__(self, path_json, root_path_im, root_path_mask, image_size):
        super(dataset_coco_mask, self).__init__()
        with open(path_json, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        data = data['annotations']
        self.files = []
        self.root_path_im = root_path_im
        self.root_path_mask = root_path_mask
        for file in data:
            name = "%012d.png"%file['image_id']
            self.files.append({'name':name, 'sentence':file['caption']})

    def __getitem__(self, idx):
        file = self.files[idx]
        name = file['name']
        # print(os.path.join(self.root_path_im, name))
        im = cv2.imread(os.path.join(self.root_path_im, name.replace('.png','.jpg')))
        im = cv2.resize(im, (512,512))
        im = img2tensor(im, bgr2rgb=True, float32=True)/255.

        mask = cv2.imread(os.path.join(self.root_path_mask, name))#[:,:,0]
        mask = cv2.resize(mask, (512,512))
        mask = img2tensor(mask, bgr2rgb=True, float32=True)[0].unsqueeze(0)#/255.

        sentence = file['sentence']
        return {'im':im, 'mask':mask, 'sentence':sentence}

    def __len__(self):
        return len(self.files)


class dataset_coco_mask_color():
    def __init__(self, path_json, root_path_im, root_path_mask, image_size):
        super(dataset_coco_mask_color, self).__init__()
        with open(path_json, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        data = data['annotations']
        self.files = []
        self.root_path_im = root_path_im
        self.root_path_mask = root_path_mask
        for file in data:
            name = "%012d.png"%file['image_id']
            self.files.append({'name':name, 'sentence':file['caption']})

    def __getitem__(self, idx):
        file = self.files[idx]
        name = file['name']
        # print(os.path.join(self.root_path_im, name))
        im = cv2.imread(os.path.join(self.root_path_im, name.replace('.png','.jpg')))
        im = cv2.resize(im, (512,512))
        im = img2tensor(im, bgr2rgb=True, float32=True)/255.

        mask = cv2.imread(os.path.join(self.root_path_mask, name))#[:,:,0]
        mask = cv2.resize(mask, (512,512))
        mask = img2tensor(mask, bgr2rgb=True, float32=True)/255.#[0].unsqueeze(0)#/255.

        sentence = file['sentence']
        return {'im':im, 'mask':mask, 'sentence':sentence}

    def __len__(self):
        return len(self.files)

class dataset_coco_mask_color_sig():
    def __init__(self, path_json, root_path_im, root_path_mask, image_size):
        super(dataset_coco_mask_color_sig, self).__init__()
        with open(path_json, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        data = data['annotations']
        self.files = []
        self.root_path_im = root_path_im
        self.root_path_mask = root_path_mask
        reg = {}
        for file in data:
            name = "%012d.png"%file['image_id']
            if name in reg:
                continue
            self.files.append({'name':name, 'sentence':file['caption']})
            reg[name] = name

    def __getitem__(self, idx):
        file = self.files[idx]
        name = file['name']
        # print(os.path.join(self.root_path_im, name))
        im = cv2.imread(os.path.join(self.root_path_im, name.replace('.png','.jpg')))
        im = cv2.resize(im, (512,512))
        im = img2tensor(im, bgr2rgb=True, float32=True)/255.

        mask = cv2.imread(os.path.join(self.root_path_mask, name))#[:,:,0]
        mask = cv2.resize(mask, (512,512))
        mask = img2tensor(mask, bgr2rgb=True, float32=True)/255.#[0].unsqueeze(0)#/255.

        sentence = file['sentence']
        return {'im':im, 'mask':mask, 'sentence':sentence, 'name': name}

    def __len__(self):
        return len(self.files)