import os
import pdb

audio_dir = './MUSIC/solo/audio'
video_dir = './MUSIC/solo/video'

all_audios = os.listdir(audio_dir)
audios = [ audio  for audio in all_audios if audio.endswith('.flac')]

all_videos = os.listdir(video_dir)
videos = [video for video in all_videos if video.endswith('.mp4')]


fid = open('solo_pairs.txt','w')
for each in audios:
    video = each.replace('.flac', '.mp4')
    if video in videos:
        fid.write(each+' '+video+'\n')
fid.close()
