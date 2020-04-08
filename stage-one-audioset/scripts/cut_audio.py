import numpy as np
import librosa
import pickle
import os
import pdb

with open('../data/audioset_single/single_train.pkl', 'rb') as fid:
    audio_list = pickle.load(fid)

audio_dir = '/mnt/scratch/hudi/audioset-instrument/audio'
video_dir = '/mnt/scratch/dongsheng/hudi/avu/instrument/data/unbalanced_train_segments_filtered_instrument_step_two/video/'

#audio_dir = '/mnt/scratch/dongsheng/hudi/avu/instrument/data/balanced_train_segments_filtered_instrument_step_two/audio'
#video_dir = '/mnt/scratch/dongsheng/hudi/avu/instrument/data/balanced_train_segments_filtered_instrument_step_two/video'
save_dir = '/mnt/scratch/hudi/audioset-instrument/audio_frames'


def audio_extract(wav_name, sr=16000):
    # pdb.set_trace()
    wav_file = os.path.join(audio_dir, wav_name)
    save_path = os.path.join(save_dir, wav_name[:-4])
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    wav, cur_sr = librosa.load(wav_file, sr=sr)
    if cur_sr != sr:
        pdb.set_trace()
    secs = int(len(wav) / sr)
    #print(secs)
    for i in range(secs):
        start = sr * i
        end = sr * (i + 1)
        cur_wav = wav[start:end]
        # spec = librosa.core.stft(cur_wav, n_fft=0.01*sr, hop_length=0.005*sr,
        #    window='hann', center=True, pad_mode='constant')
        spec = librosa.core.stft(cur_wav, n_fft=160, hop_length=80,
                                 window='hann', center=True, pad_mode='constant')
        # mel = librosa.feature.melspectrogram(S = np.abs(spec), sr=sr, n_mels=256, fmax=sr/2)
        mel = librosa.feature.melspectrogram(S=np.abs(spec), sr=sr, n_mels=64, fmax=sr / 2)
        log_mel = librosa.core.power_to_db(mel)
        log_mel_T = log_mel.T.astype('float32')
        assert log_mel_T.shape == (201, 64)
        # pdb.set_trace()
        save_name = os.path.join(save_path, '{:03d}.pkl'.format(i))
        # print(save_name)

        with open(save_name, 'wb') as fid:
            pickle.dump(log_mel_T, fid)


def find_file_using_prefix(file_prefix, path=audio_dir):
    files = os.listdir(path)
    for file_name in files:
        if file_name[:11] == file_prefix:
            return file_name

count = 0
for audio in audio_list:
    audio_name = find_file_using_prefix(audio)
    if audio_name is not None:
        if os.path.exists(os.path.join(video_dir, audio_name[:-4]+'.mp4')):
            print(audio_name)
            print(count)
            audio_extract(audio_name)
            count += 1
