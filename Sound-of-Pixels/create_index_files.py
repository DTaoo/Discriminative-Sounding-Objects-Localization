import os
import glob
import argparse
import random
import fnmatch


trainset = []
valset = []
trainfile = './data/solo_training_1.txt'
valfile = './data/solo_validation.txt'
with open(trainfile, 'r') as f:
    while True:
        line = f.readline()
        if len(line) == 0:
            break
        trainset.append(line.split('.')[0])
with open(valfile, 'r') as f:
    while True:
        line = f.readline()
        if len(line) == 0:
            break
        valset.append(line.split('.')[0])


def find_recursive(root_dir, ext='.wav'):
    files = []
    for root, dirnames, filenames in os.walk(root_dir):
        for filename in fnmatch.filter(filenames, '*' + ext):
            files.append(os.path.join(root, filename))
    return files


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_audio', default='./data/audio',
                        help="root for extracted audio files")
    parser.add_argument('--root_frame', default='./data/frames',
                        help="root for extracted video frames")
    parser.add_argument('--fps', default=4, type=int,
                        help="fps of video frames")
    parser.add_argument('--path_output', default='./data',
                        help="path to output index files")
    parser.add_argument('--trainset_ratio', default=0.8, type=float,
                        help="80% for training, 20% for validation")
    args = parser.parse_args()

    # find all audio/frames pairs
    infos = {}
    audio_files = find_recursive(args.root_audio, ext='.wav')
    for audio_path in audio_files:
        frame_path = audio_path.replace(args.root_audio, args.root_frame)[:-4]
        if not os.path.exists(frame_path):
            continue
        frame_files = os.listdir(frame_path)
        infos[frame_path.split('/')[-1]] = ','.join([audio_path, frame_path, str(len(frame_files))])
#        if len(frame_files) > args.fps * 20:
#            infos.append(','.join([audio_path, frame_path, str(len(frame_files))]))
    print('{} audio/frames pairs found.'.format(len(infos)))

    # split train/val
#    n_train = int(len(infos) * 0.8)
#    random.shuffle(infos)
#    trainset = infos[0:n_train]
#    valset = infos[n_train:]
    for name, subset in zip(['train', 'val'], [trainset, valset]):
        filename = '{}.csv'.format(os.path.join(args.path_output, name))
        with open(filename, 'w') as f:
            for item in subset:
                f.write(infos[item] + '\n')
        print('{} items saved to {}.'.format(len(subset), filename))

    print('Done!')
