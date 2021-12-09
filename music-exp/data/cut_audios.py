import numpy as np
import librosa
import pickle
import os
import pdb

with open('indicator/solo_pairs.txt','r') as fid:
    audios = [line.strip().split(' ')[0] for line in fid.readlines()]

audio_dir = './audio'
save_dir = './solo/wav_frames'

#def audio_extract(wav_name, sr=22000):
def audio_extract(wav_name, sr=16000):
    #pdb.set_trace()
    wav_file = os.path.join(audio_dir, wav_name)
    save_path = os.path.join(save_dir, wav_name[:-4])
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    wav, cur_sr = librosa.load(wav_file, sr=sr)
    if cur_sr !=sr:
        pdb.set_trace()
    secs = int(len(wav)/sr)
    print(secs)
    for i in range(secs):
        start = sr * i
        end = sr * (i+1)
        cur_wav = wav[start:end]
        assert cur_wav.shape == (sr,)
        spec = librosa.core.stft(cur_wav, n_fft=0.01*sr, hop_length=0.005*sr, 
            window='hann', center=True, pad_mode='constant')
        
        spec = librosa.core.stft(cur_wav, n_fft=160, hop_length=80, 
            window='hann', center=True, pad_mode='constant')
        
        # mel = librosa.feature.melspectrogram(S = np.abs(spec), sr=sr, n_mels=256, fmax=sr/2)
        
        mel = librosa.feature.melspectrogram(S = np.abs(spec), sr=sr, n_mels=64, fmax=sr/2)
        log_mel = librosa.core.power_to_db(mel)
        log_mel_T = log_mel.T.astype('float32')
        assert log_mel_T.shape == (201,64)
        #pdb.set_trace()
        save_name = os.path.join(save_path, '{:03d}.pkl'.format(i))
        #print(save_name)
        
        with open(save_name, 'wb') as fid:
            pickle.dump(log_mel_T, fid)

for audio in audios:
    print(audio)
    audio = audio[:-4] + 'wav'
    audio_extract(audio)
    #pdb.set_trace()
