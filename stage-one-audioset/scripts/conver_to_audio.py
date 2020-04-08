import os
import subprocess
import linecache
import pickle

with open('../data/audioset_single/single_train.pkl', 'rb') as fid:
    video_list = pickle.load(fid)

audio_dir = '/mnt/scratch/hudi/audioset-instrument/audio/'
video_dir = '/mnt/scratch/dongsheng/hudi/avu/instrument/data/unbalanced_train_segments_filtered_instrument_step_two/video/'


def find_file_using_prefix(file_prefix, path=video_dir):
    files = os.listdir(path)
    for file_name in files:
        if file_name[:11] == file_prefix:
            return file_name

for each_video in video_list:
    video_name = find_file_using_prefix(each_video)
    output_file = video_name[:-3]+'wav'

    input_file = video_dir + video_name
    output_file = audio_dir + output_file
    subprocess.call(['ffmpeg', '-i', input_file, '-ac', '-2', '-f', 'wav', output_file])