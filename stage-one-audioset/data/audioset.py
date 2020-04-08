import numpy as np
import librosa
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

class Audioset_Dataset(object):

    def __init__(self, data_root, data_list_file, opt):
        # self.root = root
        # root = '/mnt/scratch/hudi/MUSIC/solo'
        self.opt = opt
        self.audio_root = os.path.join(data_root, 'audio_frames')
        self.video_root = os.path.join(data_root, 'video_frames')

        with open(os.path.join(self.opt.data_list_dir, data_list_file), 'rb') as fid:
            data_list = pickle.load(fid)
        data_list = data_list[:100]

        with open(os.path.join(self.opt.data_list_dir, data_list_file[:-4]+'_label.pkl'), 'rb') as fid:
            self.data_label_dict = pickle.load(fid)

        self.audio_list = []
        self.video_list = []
        self.label_list = []

        for each in data_list:
            audio_path = os.path.join(self.audio_root, each)
            video_path = os.path.join(self.video_root, each)

            audio_samples = os.listdir(audio_path)
            for item in range(len(audio_samples)):
                audio_segment = audio_samples[item]
                video_segment = os.path.join(video_path, 'frame_' + audio_segment[:3])
                if os.path.exists(video_segment):
                    self.audio_list.append(os.path.join(audio_path, audio_segment))
                    self.video_list.append(os.path.join(video_path, video_segment))

        if self.opt.mode == 'val' or self.opt.mode == 'test':
            img_transform_list = [transforms.Resize((224, 224)), transforms.ToTensor()]
        else:
            img_transform_list = [transforms.Resize((256, 256)), transforms.RandomCrop(224), transforms.ToTensor()]

        self.img_transform = transforms.Compose(img_transform_list)
        # self.audio_transform = audio_transform

    def __len__(self):
        return len(self.audio_list)

    def __getitem__(self, index):

        # positive
        cur_audio_segment = self.audio_list[index]
        posi_video_segment = self.video_list[index]
        posi_video_segment_img = random.choice(os.listdir(posi_video_segment))

        # load data
        with open(cur_audio_segment, 'rb') as fid:
            cur_audio_data = pickle.load(fid)
        cur_audio_data = np.expand_dims(cur_audio_data, 0)
        posi_img_path = os.path.join(posi_video_segment, posi_video_segment_img)
        posi_img = Image.open(posi_img_path)
        if (self.opt.enable_img_augmentation and self.opt.mode == 'train'):
            posi_img = augment_image(posi_img)
        posi_img = self.img_transform(posi_img)
        posi_label = np.argmax(self.data_label_dict[posi_video_segment[-21:-10]])
        # TODO: here may need normalization

        # negative
        while (1):
            nega_video_segment = random.choice(self.video_list)
            if nega_video_segment[-11:] != posi_video_segment[-11:]:
                break
        nega_video_segment_img = random.choice(os.listdir(nega_video_segment))
        nega_img_path = os.path.join(nega_video_segment, nega_video_segment_img)
        nega_img = Image.open(nega_img_path)
        if (self.opt.enable_img_augmentation and self.opt.mode == 'train'):
            nega_img = augment_image(nega_img)
        nega_img = self.img_transform(nega_img)
        nega_label = np.argmax(self.data_label_dict[nega_video_segment[-21:-10]])

        return cur_audio_data, posi_img, nega_img, posi_label, nega_label, posi_video_segment, cur_audio_segment


class Audioset_AV_Classify(object):

    def __init__(self, video_dirs, aud_dirs, label, opt):
        self.opt = opt
        self.video_dirs = video_dirs
        self.aud_dirs = aud_dirs
        self.label = label
        if self.opt.mode == 'val' or self.opt.mode == 'test':
            img_transform_list = [transforms.Resize((224, 224)), transforms.ToTensor()]
        else:
            img_transform_list = [transforms.Resize((256, 256)), transforms.RandomCrop(224), transforms.ToTensor()]
        self.img_transform = transforms.Compose(img_transform_list)

    def __len__(self):
        return len(self.video_dirs)

    def __getitem__(self, index):
        video_segment_img = random.choice(os.listdir(self.video_dirs[index]))
        img_path = os.path.join(self.video_dirs[index], video_segment_img)

        img = Image.open(img_path)
        if (self.opt.enable_img_augmentation and self.opt.mode == 'train'):
            img = augment_image(img)
        img_data = self.img_transform(img)

        with open(self.aud_dirs[index], 'rb') as fid:
            cur_audio_data = pickle.load(fid)
        audio_data = np.expand_dims(cur_audio_data, 0)

        return audio_data, img_data, self.label[index]