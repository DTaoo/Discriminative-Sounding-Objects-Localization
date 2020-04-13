import numpy as np
import pickle
import json

#load label, cluster, and keys
labels = np.load('labels.npy')
preds = np.load('preds.npy')
keylist = pickle.load(open('keylist.pkl', 'rb'))

num_class = len(keylist)

#generate labels for duet videos
key2label = dict()
for i, k in enumerate(keylist):
    key2label[k] = i
    
duet_labels = {}
json_file = 'data/MUSIC_label/MUSIC_duet_videos.json'
duet = json.load(open(json_file))
for k in duet['videos']:
    k1, k2 = k.split(' ')
    label = np.zeros(num_class)
    label[key2label[k1]] = 1
    label[key2label[k2]] = 1
    for vid in duet['videos'][k]:
        duet_labels[vid] = label

#statistic correspondence between labels and clusterings, may need to check manually
label2cluster = np.zeros((num_class, num_class))
for i in range(num_class):
    for j in range(num_class):
        label2cluster[i, j] = np.sum(preds[labels==i]==j) / np.sum(labels==i)
label2cluster = np.argmax(label2cluster, 1)

cluster2label = np.zeros((num_class, num_class))
for i in range(num_class):
    for j in range(num_class):
        cluster2label[i, j] = np.sum(labels[preds==i]==j) / np.sum(preds==i)
cluster2label = np.argmax(cluster2label, 1)

