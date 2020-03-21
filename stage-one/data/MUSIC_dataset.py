import numpy as np
import librosa
from PIL import Image, ImageEnhance
import pickle
import random
import os
import torchvision.transforms as transforms
import json

def augment_image(image):
    if(random.random() < 0.5):
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(random.random()*0.6 + 0.7)
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(random.random()*0.6 + 0.7)
    return image


class MUSIC_Dataset(object):

    def __init__(self, data_root, data_list_file, opt):
        # self.root = root
        # root = '/mnt/scratch/hudi/MUSIC/solo'
        self.opt = opt
        self.audio_root = os.path.join(data_root, 'audio_frames')
        self.video_root = os.path.join(data_root, 'video_frames')

        with open(data_list_file,'r') as fid:
            pairs = [line.strip().split(' ') for line in fid.readlines()]
       
        self.sample_label= self._parse_csv(self.opt.json_file)

        self.audio_list = []
        self.video_list = []
        self.label_list = []

        for each in pairs:
            audio = each[0]
            video = each[1]
            assert audio[:-5] == video[:-4]
            audio_path = os.path.join(self.audio_root, audio[:-5])
            video_path = os.path.join(self.video_root, video[:-4])

            audio_samples= os.listdir(audio_path)
            for item in range(len(audio_samples)):
                audio_segment = audio_samples[item]
                video_segment = os.path.join(video_path, 'frame_'+audio_segment[:3])
                if os.path.exists(video_segment):
                    self.audio_list.append(os.path.join(audio_path, audio_segment))
                    self.video_list.append(os.path.join(video_path, video_segment))

        if self.opt.mode == 'val' or self.opt.mode == 'test':
            img_transform_list = [transforms.Resize((224,224)), transforms.ToTensor()]
        else:
            img_transform_list = [transforms.Resize((256, 256)), transforms.RandomCrop(224), transforms.ToTensor()]

        self.img_transform = transforms.Compose(img_transform_list)
        #self.audio_transform = audio_transform

    def __len__(self):
        return len(self.audio_list)


    def _parse_csv(self, json_file):
        f = open(json_file, encoding='utf-8')
        content = f.read()
        ins_indicator = json.loads(content)
        ins_indicator = ins_indicator['videos']
        ins_list = [*ins_indicator]
        sample_label = {}
        for i in range(len(ins_list)):
            current_list = ins_indicator[ins_list[i]]
            for j in range(len(current_list)):
                sample_label[current_list[j]] = i
        return sample_label


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
        if(self.opt.enable_img_augmentation and self.opt.mode == 'train'):
            posi_img = augment_image(posi_img)
        posi_img = self.img_transform(posi_img)
        #print(posi_video_segment[-38:-20])        
        #print(posi_video_segment)
        posi_label = self.sample_label[posi_video_segment[-28:-17]]
        # TODO: here may need normalization


        # negative
        while(1):
            nega_video_segment = random.choice(self.video_list)
            #nega_label = self.sample_label[nega_video_segment[-28:-17]]
            if nega_video_segment[-28:-17] != posi_video_segment[-28:-17]:
            #if posi_label != nega_label:
                break
        nega_video_segment_img = random.choice(os.listdir(nega_video_segment))
        nega_img_path = os.path.join(nega_video_segment, nega_video_segment_img)
        nega_img = Image.open(nega_img_path)       
        if(self.opt.enable_img_augmentation and self.opt.mode == 'train'):
            nega_img = augment_image(nega_img)
        nega_img = self.img_transform(nega_img)
        nega_label = self.sample_label[nega_video_segment[-28:-17]]

        #print('%d %d' %(posi_label,nega_label))
        #print(cur_audio_data.shape)
        return cur_audio_data, posi_img, nega_img, posi_label, nega_label, posi_video_segment+'/'+posi_video_segment_img


class MUSIC_Dataset_video(object):

    def __init__(self, data_root, data_list_file, opt):
        # self.root = root
        # root = '/mnt/scratch/hudi/MUSIC/solo'
        self.opt = opt
        self.audio_root = os.path.join(data_root, 'audio_frames')
        self.video_root = os.path.join(data_root, 'video_frames')
        self.sample_label = self._parse_csv(self.opt.json_file)

        with open(data_list_file,'r') as fid:
            pairs = [line.strip().split(' ') for line in fid.readlines()]

        self.audio_list = []
        self.video_list = []
        self.label_list = []

        for each in pairs:
            audio = each[0]
            video = each[1]
            assert audio[:-5] == video[:-4]
            audio_path = os.path.join(self.audio_root, audio[:-5])
            video_path = os.path.join(self.video_root, video[:-4])

            audio_samples= os.listdir(audio_path)
            for item in range(len(audio_samples)):
                audio_segment = audio_samples[item]
                video_segment = os.path.join(video_path, 'frame_'+audio_segment[:3])
                if os.path.exists(video_segment):
                    self.audio_list.append(os.path.join(audio_path, audio_segment))
                    self.video_list.append(os.path.join(video_path, video_segment))

        if self.opt.mode == 'val' or self.opt.mode == 'test':
            img_transform_list = [transforms.Resize((224,224)), transforms.ToTensor()]
        else:
            img_transform_list = [transforms.Resize((256, 256)), transforms.RandomCrop(224), transforms.ToTensor()]

        self.img_transform = transforms.Compose(img_transform_list)
        #self.audio_transform = audio_transform

    def __len__(self):
        return len(self.audio_list)

    def _get_video_frames(self, video_dir):
        frame_list = os.listdir(video_dir)
        assert len(frame_list) == 4
        frames = np.zeros(shape=(3, 4, 224, 224))
        for i in range(4):
            img_path = os.path.join(video_dir, frame_list[i])
            img = Image.open(img_path)
            if (self.opt.enable_img_augmentation and self.opt.mode == 'train'):
                img = augment_image(img)
            img = self.img_transform(img)
            #img = np.expand_dims(img, 1)
            #print(img.shape)
            frames[:,i,:,:] = img
        return frames

    def _parse_csv(self, json_file):
        f = open(json_file, encoding='utf-8')
        content = f.read()
        ins_indicator = json.loads(content)
        ins_indicator = ins_indicator['videos']
        ins_list = [*ins_indicator]
        sample_label = {}
        for i in range(len(ins_list)):
            current_list = ins_indicator[ins_list[i]]
            for j in range(len(current_list)):
                sample_label[current_list[j]] = i
        return sample_label

    def __getitem__(self, index):

        # positive
        cur_audio_segment = self.audio_list[index]
        posi_video_segment = self.video_list[index]

        # load data
        with open(cur_audio_segment, 'rb') as fid:
            cur_audio_data = pickle.load(fid)
        cur_audio_data = np.expand_dims(cur_audio_data, 0)
        posi_img = self._get_video_frames(posi_video_segment)
        posi_label = self.sample_label[posi_video_segment[-28:-17]]

        # TODO: here may need normalization

        # negative
        while(1):
            nega_video_segment = random.choice(self.video_list)
            if nega_video_segment[-28:-17] != posi_video_segment[-28:-17]:
                break
        nega_img = self._get_video_frames(nega_video_segment)
        nega_label = self.sample_label[nega_video_segment[-28:-17]]

        #print(cur_audio_data.shape)
        return cur_audio_data, posi_img, nega_img, posi_label, nega_label, posi_video_segment


class MUSIC_Classify_Data(object):

    def __init__(self, img_dirs, label, opt):
        self.opt = opt
        self.img_dirs = img_dirs
        self.label = label
        if self.opt.mode == 'val' or self.opt.mode == 'test':
            img_transform_list = [transforms.Resize((224,224)), transforms.ToTensor()]
        else:
            img_transform_list = [transforms.Resize((256, 256)), transforms.RandomCrop(224), transforms.ToTensor()]
        self.img_transform = transforms.Compose(img_transform_list)

    def __len__(self):
        return len(self.img_dirs)


    def __getitem__(self, index):

        img = Image.open(self.img_dirs[index])
        if (self.opt.enable_img_augmentation and self.opt.mode == 'train'):
            img = augment_image(img)
        img = self.img_transform(img)

        return img, self.label[index]


class MUSIC_Video_Classify(object):

    def __init__(self, video_dirs, label, opt):
        self.opt = opt
        self.video_dirs = video_dirs
        self.label = label
        if self.opt.mode == 'val' or self.opt.mode == 'test':
            img_transform_list = [transforms.Resize((224,224)), transforms.ToTensor()]
        else:
            img_transform_list = [transforms.Resize((256, 256)), transforms.RandomCrop(224), transforms.ToTensor()]
        self.img_transform = transforms.Compose(img_transform_list)

    def __len__(self):
        return len(self.video_dirs)

    def _get_video_frames(self, video_dir):
        frame_list = os.listdir(video_dir)
        assert len(frame_list) == 4
        frames = np.zeros(shape=(3, 4, 224, 224))
        for i in range(4):
            img_path = os.path.join(video_dir, frame_list[i])
            img = Image.open(img_path)
            if (self.opt.enable_img_augmentation and self.opt.mode == 'train'):
                img = augment_image(img)
            img = self.img_transform(img)
            #img = np.expand_dims(img, 1)
            #print(img.shape)
            frames[:,i,:,:] = img
        return frames

    def __getitem__(self, index):
        imgs = self._get_video_frames(self.video_dirs[index])
        return imgs, self.label[index]

# # for a given audio clip, we mush confirm its length: len(audio) = sr


# def audio_extract(wav_file, sr=22000):
#     wav = librosa.load(wav_file, sr=sr)[0]
#     # Takes a waveform (length 160,000, sampling rate 16,000) and extracts filterbank features (size 400 * 64)
#     spec = librosa.core.stft(wav, n_fft = 0.01*sr, hop_length =0.005*sr,
#                              window = 'hann', center = True, pad_mode = 'constant')
#     mel = librosa.feature.melspectrogram(S = np.abs(spec), sr = sr, n_mels = 256, fmax = sr/2)
#     logmel = librosa.core.power_to_db(mel)

#     return logmel.T.astype('float32')


# def augment_image(image):
#     if(random.random() < 0.5):
#         image = image.transpose(Image.FLIP_LEFT_RIGHT)
#     enhancer = ImageEnhance.Brightness(image)
#     image = enhancer.enhance(random.random()*0.6 + 0.7)
#     enhancer = ImageEnhance.Color(image)
#     image = enhancer.enhance(random.random()*0.6 + 0.7)
#     return image


