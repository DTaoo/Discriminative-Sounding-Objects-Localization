import pickle
import json
import cv2
import sklearn.metrics
import numpy as np
import os

#load bounding box and predicted avmap result
# gt = json.load(open('duettest.json', 'r'))
pred = pickle.load(open('syn_objs.pkl', 'rb'))
indexs = np.load('corr.npy')
corr = np.argmax(indexs, 1)
keylist = ['flute', 'acoustic_guitar', 'accordion', 'xylophone', 'erhu', 'tuba', 
           'saxophone', 'cello', 'violin', 'clarinet', 'trumpet']
keylist.sort()

duetkeys = {13: 'accordion', 3: 'acoustic_guitar', 1: 'cello', 12: 'flute', 8: 'saxophone', 
            10: 'trumpet', 11: 'violin'}

cious = []
nosounds = []
thres = 0.5#threshold to determine whether positive on avmap
visualize = 0

for k in pred:
    gt = np.load(os.path.join('/home/ruiq/Music/synthetic/test1/box', k+'.npy'))
    boxs = {}
    for box in gt:
        category = box[-1]
        index = keylist.index(duetkeys[int(category)])
        index = corr[index]
        if index not in boxs:
            boxs[index] = []
        boxs[index].append(box[:4])

    thresvalue = 0.2
    # thresvalue = thres * np.max(pred[k])

    ciou = []
    nosound = []
    for idx in range(11):
        '''
        img = os.path.join('/home/ruiq/Music/synthetic/test1/video', k+'.jpg')
        img = cv2.imread(img)
        img = cv2.resize(img, (224, 224))
        predmap = cv2.resize(pred[k][idx], (224, 224))
        if visualize:
            # predmap[predmap>thresvalue] = 0.8
            # predmap[predmap<0.8] = 0
            predmap = np.uint8(255 * predmap)
            predmap = cv2.applyColorMap(predmap, cv2.COLORMAP_JET)
            img = 0.6 * img + 0.4 * predmap
            cv2.imwrite('visualize/stage2/'+k+'_%d.jpg'%idx, img)
'''
        if idx in boxs:
            continue
        predmap = cv2.resize(pred[k][idx], (224, 224))
        nosound.append(np.sum(predmap<=thresvalue)/(224*224))
        
    for idx in boxs:
        # img = os.path.join('/home/ruiq/Music/synthetic/test1/video', k+'.jpg')
        # img = cv2.imread(img)
        # img = cv2.resize(img, (224, 224))
        gtmap = np.zeros((224, 224))
        for i in range(len(boxs[idx])):
            box = boxs[idx][i]
            box = np.array(box) * 224
            gtmap[int(box[1]): int(box[1]+box[3]), int(box[0]): int(box[0]+box[2])] = 1
        #resize predited avmap to (224, 224)
        row = indexs[np.where(corr==idx)[0][0]]
        predmap = cv2.resize(np.sum(pred[k][row==1], 0), (224, 224))
        #calculate ciou
        thresvalue = thres * np.max(predmap)
        ciou.append(np.sum((predmap>thresvalue) * (gtmap>0)) / 
                    (np.sum(gtmap) + np.sum((predmap>thresvalue) * (gtmap==0))))
        
        if visualize:
            # predmap[predmap>thresvalue] = 0.8
            # predmap[predmap<0.8] = 0
            predmap = np.uint8(255 * predmap)
            predmap = cv2.applyColorMap(predmap, cv2.COLORMAP_JET)
            img = 0.6 * img + 0.4 * predmap
            cv2.rectangle(img, (int(box[0]), int(box[1])), 
                          (int(box[0]+box[2]), int(box[1]+box[3])), (0, 255, 0), 2)
            cv2.imwrite('visualize/stage2/'+k+'_%d.jpg'%idx, img)
        
    ciou = np.mean(ciou)
    cious.append(ciou)
    nosounds.append(np.mean(nosound))
    
results = []
for i in range(21):
    result = np.sum(np.array(cious) >= 0.05 * i)
    result = result / len(cious)
    results.append(result)
x = [0.05 * i for i in range(21)]
auc = sklearn.metrics.auc(x, results)
print(auc, results[6], results[10], np.mean(nosounds))
# '''
