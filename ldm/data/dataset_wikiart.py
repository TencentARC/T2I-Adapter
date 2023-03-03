import json
import os.path

from PIL import Image
from torch.utils.data import DataLoader

from transformers import CLIPProcessor
from torchvision.transforms import transforms

import pytorch_lightning as pl


class WikiArtDataset():
    def __init__(self, meta_file):
        super(WikiArtDataset, self).__init__()

        self.files = []
        with open(meta_file, 'r') as f:
            js = json.load(f)
            for img_path in js:
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                caption = img_name.split('_')[-1]
                caption = caption.split('-')
                j = len(caption) - 1
                while j >= 0:
                    if not caption[j].isdigit():
                        break
                    j -= 1
                if j < 0:
                    continue
                sentence = ' '.join(caption[:j + 1])
                self.files.append({'img_path': os.path.join('datasets/wikiart', img_path), 'sentence': sentence})

        version = 'openai/clip-vit-large-patch14'
        self.processor = CLIPProcessor.from_pretrained(version)

        self.jpg_transform = transforms.Compose([
            transforms.Resize(512),
            transforms.RandomCrop(512),
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx):
        file = self.files[idx]

        im = Image.open(file['img_path'])

        im_tensor = self.jpg_transform(im)

        clip_im = self.processor(images=im, return_tensors="pt")['pixel_values'][0]

        return {'jpg': im_tensor, 'style': clip_im, 'txt': file['sentence']}

    def __len__(self):
        return len(self.files)


class WikiArtDataModule(pl.LightningDataModule):
    def __init__(self, meta_file, batch_size, num_workers):
        super(WikiArtDataModule, self).__init__()
        self.train_dataset = WikiArtDataset(meta_file)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                          pin_memory=True)
