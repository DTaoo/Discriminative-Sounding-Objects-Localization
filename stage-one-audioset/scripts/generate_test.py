import pickle
import os
import subprocess

with open('../data/audioset_single/single_train.pkl', 'rb') as fid:
    video_list = pickle.load(fid)


audio_dir = '/mnt/scratch/hudi/audioset-instrument/audio_frames'
video_dir = '/mnt/scratch/hudi/audioset-instrument/video_frames'

def find_file_using_prefix(file_prefix, path=audio_dir):
    files = os.listdir(path)
    for file_name in files:
        if file_name[:11] == file_prefix:
            return file_name


for i in range(100):
    cur_video = video_list[i]
    cur_audio = find_file_using_prefix(cur_video, path=audio_dir)
    cur_image = find_file_using_prefix(cur_video, path=video_dir)
    try:
        subprocess.call(['mv', os.path.join(audio_dir, cur_audio), os.path.join(audio_dir, cur_audio[:11])])
        subprocess.call(['mv', os.path.join(video_dir, cur_image), os.path.join(video_dir, cur_image[:11])])
    except:
        print(cur_audio)
