import numpy as np
import pickle
import json

# load label, cluster, and keys
# labels = np.load('labels.npy')
# preds = np.load('preds.npy')
# labels = pickle.load(open('label.pkl', 'rb'))
# preds = pickle.load(open('cluster_label.pkl', 'rb'))
cluster = pickle.load(open('obj_features2/cluster_6.pkl', 'rb'))
labels = cluster['gt_labels']
preds = cluster['pseudo_label']
keylist = ['Banjo', 'Cello', 'Drum', 'Guitar', 'Harp', 'Harmonica', 'Piano', 'Saxophone', 
           'Trombone', 'Violin', 'Flute', 'Accordion', 'Horn']

num_class = len(keylist)

#generate labels for duet videos
key2label = dict()
for i, k in enumerate(keylist):
    key2label[k] = i
    
#statistic correspondence between labels and clusterings, may need to check manually
label2cluster = np.zeros((num_class, num_class))
for i in range(num_class):
    for j in range(num_class):
        label2cluster[i, j] = np.sum(preds[labels==i]==j) / np.sum(labels==i)
#print(np.max(label2cluster, 1))
l2c = np.argmax(label2cluster, 1)

cluster2label = np.zeros((num_class, num_class))
for i in range(num_class):
    for j in range(num_class):
        cluster2label[i, j] = np.sum(labels[preds==i]==j) / np.sum(preds==i)
c2l = np.argmax(cluster2label, 1)

