import pickle
import json
import cv2
import sklearn.metrics
import numpy as np

def cal_ciou_class():
    #load bounding box and predicted avmap result
    gt = json.load(open('musictest.json', 'r'))
    pred = pickle.load(open('multi.pkl', 'rb'))
    cious = []
    nosounds = []
    thres = 0.0001 #threshold to determine whether positive on avmap
    nosound_thres = 1/13 #nosound threshold 1/13 for audioset and 1/11 for music
    
    for k in pred:
        if k not in gt:
            continue
        if len(gt[k]['bbox']) == 0:
            continue
                
        boxs = {}
        for i in range(len(gt[k]['bbox'])):
            box = gt[k]['bbox'][i]
            category = box['category']
            if category not in boxs:
                boxs[category] = []
            boxs[category].append(box['normbox'])
            
        #resize predited avmap to (224, 224)
        predmap = cv2.resize(pred[k][0], (224, 224))
        nosound = np.sum(predmap<=nosound_thres) / (224*224)
      
        ciou = []
        for i in boxs:
            gtmap = np.zeros((224, 224))
            for box in boxs[i]:
                box = np.array(box) * 224
                #calculate groundtruth map of size (224, 224)
                gtmap[int(box[1]): int(box[1]+box[3]), int(box[0]): int(box[0]+box[2])] = 1
            #calculate ciou
            iou = np.sum((predmap>thres*np.max(predmap)) * (gtmap>0)) / (np.sum(gtmap) + np.sum((predmap>thres*np.max(predmap)) * (gtmap==0)))
            ciou.append(iou)
        ciou = np.mean(ciou)
        cious.append(ciou)
        nosounds.append(nosound)
        
    results = []
    for i in range(21):
        result = np.sum(np.array(cious) >= 0.05 * i)
        result = result / len(cious)
        results.append(result)
    x = [0.05 * i for i in range(21)]
    auc = sklearn.metrics.auc(x, results)
    print('AUC %.3f , CIOU@0.5 %.3f , average nosound %.3f'%(auc, results[10], np.mean(nosounds)))
