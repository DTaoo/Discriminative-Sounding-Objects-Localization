import os
import cv2
import pdb

def video2frame(video_path, frame_save_path, frame_interval=1):

    vid = cv2.VideoCapture(video_path)
    fps = vid.get(cv2.CAP_PROP_FPS)
    #pdb.set_trace()
    success, image = vid.read()
    count = 0
    while success:
        count +=1
        if count % frame_interval == 0:
            #cv2.imencode('.png', image)[1].tofile(frame_save_path+'/fame_%d.png'%count)
            save_name = '{}/frame_{}_{}.jpg'.format(frame_save_path, int(count/fps),count)
            cv2.imencode('.jpg', image)[1].tofile(save_name)
        success, image = vid.read()
    print(count)


def video2frame_update(video_path, frame_save_path, frame_kept_per_second=4):

    vid = cv2.VideoCapture(video_path)
    fps = vid.get(cv2.CAP_PROP_FPS)
    video_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    video_len = int(video_frames/fps)
    print(video_len)

    count = 0
    frame_interval = int(fps/frame_kept_per_second)
    while(count < fps*video_len):
        ret, image = vid.read()
        if not ret:
            break
        if count % fps == 0:
            frame_id = 0
        if frame_id<frame_interval*frame_kept_per_second and frame_id%frame_interval == 0:
            #cv2.imencode('.png', image)[1].tofile(frame_save_path+'/fame_%d.png'%count)
            save_dir = '{}/frame_{:03d}'.format(frame_save_path, int(count/fps))
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            save_name = '{}/frame_{:03d}/{:05d}.jpg'.format(frame_save_path, int(count/fps), count)
            cv2.imencode('.jpg', image)[1].tofile(save_name)
        
        frame_id += 1
        count += 1




video_dir = './MUSIC/solo/video'
#videos = os.listdir(video_dir)
with open('data_indicator/music/solo/solo_pairs.txt','r') as fid:
    videos = [line.strip().split(' ')[1] for line in fid.readlines()]
save_dir = './MUSIC/solo/video_frames'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
vid_count = 0
for each_video in videos:
    if not each_video.endswith('.mp4'):
        continue
    print(each_video)
    video_path = os.path.join(video_dir, each_video)
    save_path = os.path.join(save_dir, each_video[:-4])
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    video2frame_update(video_path, save_path, frame_kept_per_second=4)
    #pdb.set_trace()
print('cut %d videos' % vid_count)
