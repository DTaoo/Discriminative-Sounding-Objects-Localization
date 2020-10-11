import numpy as np
import torch
from PIL import Image, ImageEnhance
import pickle
import random
import os
import torchvision.transforms as transforms
import json


def augment_image(image):
    if (random.random() < 0.5):
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(random.random() * 0.6 + 0.7)
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(random.random() * 0.6 + 0.7)
    return image


class MUSIC_Dataset(object):

    def __init__(self, opt):
        # self.root = root
        # root = '/mnt/scratch/hudi/MUSIC/solo'
        self.opt = opt
        if self.opt.mode == 'train':
            self.audio_root = '/home/ruiq/Music/synthetic/train/train/audio'
            self.video_root = '/home/ruiq/Music/synthetic/train/train/video'
        else:
            self.audio_root = '/home/ruiq/Music/synthetic/test1/audio'
            self.video_root = '/home/ruiq/Music/synthetic/test1/video'
        self.box_root = '/home/ruiq/Music/synthetic/test1/box'

        self.audio_list = os.listdir(self.audio_root)
        self.video_list = os.listdir(self.video_root)
        self.box_list = os.listdir(self.box_root)
        self.audio_list.sort()
        self.video_list.sort()
        self.box_list.sort()

        assert len(self.audio_list) == len(self.video_list)

        if self.opt.mode == 'val' or self.opt.mode == 'test':
            img_transform_list = [transforms.Resize((224, 224)), transforms.ToTensor(),
                                  transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
        else:
            img_transform_list = [transforms.Resize((224, 224)), transforms.ToTensor(),
                                  transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]

        self.img_transform = transforms.Compose(img_transform_list)

    def __len__(self):
        return len(self.audio_list)

    def __getitem__(self, index):

        # positive
        cur_audio_segment = self.audio_list[index]
        posi_video_segment = self.video_list[index]
        if self.opt.mode == 'val':
            box_segment = self.box_list[index]

        # load data
        with open(os.path.join(self.audio_root, cur_audio_segment), 'rb') as fid:
            cur_audio_data = pickle.load(fid)
        cur_audio_data = np.expand_dims(cur_audio_data, 0)
        posi_img_path = os.path.join(self.video_root, posi_video_segment)
        posi_img = Image.open(posi_img_path)
        if (self.opt.enable_img_augmentation and self.opt.mode == 'train'):
            posi_img = augment_image(posi_img)
        posi_img = self.img_transform(posi_img)

        while (1):
            nega_video_segment = random.choice(self.video_list)
            if nega_video_segment != posi_video_segment:
                break
        nega_img_path = os.path.join(self.video_root, nega_video_segment)
        nega_img = Image.open(nega_img_path)
        if (self.opt.enable_img_augmentation and self.opt.mode == 'train'):
            nega_img = augment_image(nega_img)
        nega_img = self.img_transform(nega_img)

        if self.opt.mode == 'val':
            box = np.load(os.path.join(self.box_root, box_segment))
            return cur_audio_data, posi_img, nega_img, box
        return cur_audio_data, posi_img, nega_img
