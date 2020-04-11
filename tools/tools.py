import pickle
import json
import cv2
import sklearn.metrics
import numpy as np
import matplotlib.pyplot as plt

def cal_ciou():
    #load bounding box and predicted avmap result
    gt = json.load(open('musictest.json', 'r'))
    pred = pickle.load(open('single.pkl', 'rb'))
    cious = []
    thres = 0.0001 #threshold to determine whether positive on avmap
    
    for k in pred:
        if k not in gt:
            continue
        if len(gt[k]['bbox']) == 0:
            continue
        
        box = gt[k]['bbox'][0]['normbox']
        box = np.array(box) * 224
        #calculate groundtruth map of size (224, 224)
        gtmap = np.zeros((224, 224))
        gtmap[int(box[1]): int(box[1]+box[3]), int(box[0]): int(box[0]+box[2])] = 1
        #resize predited avmap to (224, 224)
        predmap = cv2.resize(pred[k][0], (224, 224))
        #calculate ciou
        ciou = np.sum((predmap>thres) * (gtmap>0)) / (np.sum(gtmap) + np.sum((predmap>thres) * (gtmap==0)))
        cious.append(ciou)
        
    results = []
    for i in range(21):
        result = np.sum(np.array(cious) >= 0.05 * i)
        result = result / len(cious)
        results.append(result)
    x = [0.05 * i for i in range(21)]
    auc = sklearn.metrics.auc(x, results)
    print(auc, results[10])


def visualize(images, cams, e):
    #images (batchsize, h, w, 3)
    #cams (batchsize, 1, h, w)
    images = images.cpu().numpy()
    cams = cams.detach().cpu().numpy()
    for i in range(images.shape[0]):
        if i % 2 == 0:
            cor = '_cor'
        else:
            cor = '_not'
        cam = cams[i, 0]
        image = images[i]
        #recover image to 0-1
        image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        image = np.clip(image, 0, 1)
        #generate heatmap
        cam = cam * 255
        cam = cv2.applyColorMap(cam.astype(np.uint8), cv2.COLORMAP_JET)
        cam = cam[:, :, ::-1] / 255
        plt.imsave('vis/img_'+str(e)+'_'+str(i)+cor+'.jpg', 0.5*cam+0.5*image)
