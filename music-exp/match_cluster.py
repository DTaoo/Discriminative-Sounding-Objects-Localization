import numpy as np
import pickle
import json

# load label, cluster, and keys
# labels = np.load('labels.npy')
# preds = np.load('preds.npy')
# labels = pickle.load(open('label.pkl', 'rb'))
# preds = pickle.load(open('cluster_label.pkl', 'rb'))
cluster = pickle.load(open('obj_features_0.05_20/cluster_10.pkl', 'rb'))
labels = cluster['gt_labels']
preds = cluster['pseudo_label']
keylist = ['flute', 'acoustic_guitar', 'accordion', 'xylophone', 'erhu', 'tuba', 
           'saxophone', 'cello', 'violin', 'clarinet', 'trumpet']
keylist.sort()

num_cluster = 20
num_class = len(keylist)

#generate labels for duet videos
key2label = dict()
for i, k in enumerate(keylist):
    key2label[k] = i
    
#statistic correspondence between labels and clusterings, may need to check manually
label2cluster = np.zeros((num_class, num_cluster))
for i in range(num_class):
    for j in range(num_cluster):
        label2cluster[i, j] = np.sum(preds[labels==i]==j) / np.sum(labels==i)
print(np.max(label2cluster, 1))

maxscore = 0
for _ in range(1000000):
    indexs = np.zeros((num_class, num_cluster))
    index = np.random.permutation(num_cluster)
    indexs[np.arange(num_class), index[:num_class]] = 1
    for i in range(num_cluster-num_class):
        indexs[np.random.randint(0, num_class), index[num_class-num_cluster+i]] = 1
    score = np.sum(indexs * label2cluster)
    if score > maxscore:
        maxscore = score
        maxindex = indexs
    
label2cluster = np.argmax(label2cluster, 1)
print(label2cluster)
np.save('corr', maxindex)

cluster2label = np.zeros((num_cluster, num_class))
for i in range(num_cluster):
    for j in range(num_class):
        cluster2label[i, j] = np.sum(labels[preds==i]==j) / np.sum(preds==i)
# cluster2label = np.argmax(cluster2label, 1)

