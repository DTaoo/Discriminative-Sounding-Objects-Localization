import pickle
import json
import cv2
import sklearn.metrics
import numpy as np
import os

#load bounding box and predicted avmap result
gt = json.load(open('duettest.json', 'r'))
pred = pickle.load(open('syn_objs.pkl', 'rb'))
avmaps = pickle.load(open('duet_avmaps.pkl', 'rb'))
indexs = np.load('corr.npy')
corr = np.argmax(indexs, 1)
keylist = ['flute', 'acoustic_guitar', 'accordion', 'xylophone', 'erhu', 'tuba', 
           'saxophone', 'cello', 'violin', 'clarinet', 'trumpet']
duetkeys = {13: 'accordion', 3: 'acoustic_guitar', 1: 'cello', 12: 'flute', 8: 'saxophone', 
            10: 'trumpet', 11: 'violin'}
keylist.sort()

cious = []
nosounds = []
unders = []
thres = 0.2#threshold to determine whether positive on avmap

visualize = 0
keys = list(pred.keys())
keys.sort()
for id, k in enumerate(keys):
# for id in [2695, 2811]:
    # k = keys[id]
    if k not in gt:
        continue
    if len(gt[k]['bbox']) == 0:
        continue
        
    boxs = {}
    for box in gt[k]['bbox']:
        category = box['category_id']
        index = keylist.index(duetkeys[category])
        index = corr[index]
        if index not in boxs:
            boxs[index] = []
        boxs[index].append(box['normbox'])
        
    nosound = []
    thresvalue = 0.2
    # thresvalue = thres * np.max(pred[k])
    
    # img = os.path.join('/home/ruiq/Music/duet/duet/video_frames', k)
    # img = cv2.imread(img)
    # img = cv2.resize(img, (224, 224))
    
    for idx in range(len(pred[k])):
        # predmap = cv2.resize(pred[k][idx], (224, 224))
        # predmap = np.uint8(255 * predmap)
        # predmap = cv2.applyColorMap(predmap, cv2.COLORMAP_JET)
        # img_ = 0.6 * img + 0.4 * predmap
        # cv2.imwrite('visualize/stage2/'+str(id)+'_%d.jpg'%idx, img_)

        # '''
        if idx in boxs:
            continue
        predmap = cv2.resize(pred[k][idx], (224, 224))
        nosound.append(np.sum(predmap<=thresvalue) / (224*224))
    nosound = np.mean(nosound)
    nosounds.append(nosound)
    
    ciou = []
    under = []
    for idx in boxs:
        # img = os.path.join('/home/ruiq/Music/duet/duet/video_frames', k)
        # img = cv2.imread(img)
        # img = cv2.resize(img, (224, 224))
        # avmap = avmaps[k]
        # avmap = cv2.resize(avmap[0], (224, 224))
        # avmap = np.uint8(avmap * 255)
        # avmap = cv2.applyColorMap(avmap, cv2.COLORMAP_JET)
        # cv2.imwrite('visualize/stage2/'+str(id)+'.jpg', 0.6*img+0.4*avmap)
        # break

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
        under.append(np.sum(predmap<=thresvalue)/(224*224))
        ciou.append(np.sum((predmap>thresvalue) * (gtmap>0)) / (np.sum(gtmap) + np.sum((predmap>thresvalue) * (gtmap==0))))
        
        if visualize:
            predmap = np.uint8(255 * predmap)
            predmap = cv2.applyColorMap(predmap, cv2.COLORMAP_JET)
            img = 0.6 * img + 0.4 * predmap
            cv2.rectangle(img, (int(box[0]), int(box[1])), 
                          (int(box[0]+box[2]), int(box[1]+box[3])), 
                          (0, 255, 0), 2)
            cv2.imwrite('visualize/stage2/'+str(id)+'_%d.jpg'%idx, img)
    ciou = np.mean(ciou)
    cious.append(ciou)
    unders.append(np.mean(under))
    
results = []
for i in range(21):
    result = np.sum(np.array(cious) >= 0.05 * i)
    result = result / len(cious)
    results.append(result)
x = [0.05 * i for i in range(21)]
auc = sklearn.metrics.auc(x, results)
print(auc, results[6], results[10], np.mean(nosounds))
# '''
