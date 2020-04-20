import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
from data.MUSIC_dataset import MUSIC_Dataset, MUSIC_AV_Classify

from model.base_model import resnet18
from model.location_model import Location_Net

from sklearn import cluster, metrics
import numpy as np
from sklearn.preprocessing import normalize
from progress.bar import Bar
import matplotlib.pyplot as plt
import cv2
import pickle


def batch_organize(audio_data, posi_img_data, nega_img_data, posi_label, nega_label):
    batch_audio_data = torch.zeros(audio_data.shape[0] * 2, audio_data.shape[1], audio_data.shape[2],
                                   audio_data.shape[3])
    batch_image_data = torch.zeros(posi_img_data.shape[0] * 2, posi_img_data.shape[1], posi_img_data.shape[2],
                                   posi_img_data.shape[3])
    batch_labels = torch.zeros(audio_data.shape[0] * 2)
    class_labels = torch.zeros(audio_data.shape[0] * 2)
    for i in range(audio_data.shape[0]):
        batch_audio_data[i * 2, :] = audio_data[i, :]
        batch_audio_data[i * 2 + 1, :] = audio_data[i, :]
        batch_image_data[i * 2, :] = posi_img_data[i, :]
        batch_image_data[i * 2 + 1, :] = nega_img_data[i, :]
        batch_labels[i * 2] = 1
        batch_labels[i * 2 + 1] = 0
        class_labels[i * 2] = posi_label[i]
        class_labels[i * 2 + 1] = nega_label[i]
    return batch_audio_data, batch_image_data, batch_labels, class_labels


def eva_metric(predict, gt):
    correct = (np.round(predict) == gt).sum()
    if np.sum(np.round(predict) == 1) == 0:
        recall = 0
    else:
        recall = np.sum((np.round(predict) == 1) * (gt == 1)) / np.sum(np.round(predict) == 1)
    return correct / predict.shape[0], recall

def location_dilation(location_map):
    location_map_dilated = np.uint8(np.zeros_like(location_map))
    for i in range(location_map.shape[1]):
        for j in range(location_map.shape[2]):
            if location_map[0, i,j] == 1:
                location_map_dilated[0, i, j] = 1
                if i > 0 and j > 0 and i < location_map.shape[1]-1 and  j < location_map.shape[2]-1:
                    location_map_dilated[0, i+1, j] = 1
                    location_map_dilated[0, i-1, j] = 1
                    location_map_dilated[0, i, j-1] = 1
                    location_map_dilated[0, i, j+1] = 1
    return location_map_dilated

def location_model_train(model, data_loader, optimizer, criterion):
    model.train()
    accs = 0
    count = 0
    losses = 0
    obj_labels = []
    obj_preds = []
    bar = Bar('processing', max=len(data_loader))
    for i, data in enumerate(data_loader, 0):
        audio_data, posi_img_data, nega_img_data, posi_label, nega_label, _ , _= data
        audio_data, image_data, av_labels, class_label = batch_organize(audio_data, posi_img_data, nega_img_data, posi_label,
                                                              nega_label)

        audio_data, image_data, av_labels = audio_data.type(torch.FloatTensor).cuda(), \
                                            image_data.type(torch.FloatTensor).cuda(), \
                                            av_labels.type(torch.FloatTensor).cuda()

        optimizer.zero_grad()
        av_outputs, _, _, _, _, audio_class = model(image_data, audio_data)
        loss = criterion(av_outputs, av_labels)
        loss.backward()
        optimizer.step()

        losses += loss.detach().cpu().numpy()
        acc, _ = eva_metric(av_outputs.detach().cpu().numpy(), av_labels.cpu().numpy())
        accs += acc
        count += 1
        
        for j in range(av_outputs.shape[0]):
            if av_labels[j].cpu().numpy() == 0:
                continue
            obj_labels.append(class_label[j].numpy())
            obj_preds.append(np.argmax(audio_class[j].detach().cpu().numpy()))
            
        bar.suffix  = '({batch}/{size}) Loss: {loss:.3f} |Acc: {acc:.3f}'.format(
            batch=i + 1,
            size=len(data_loader),
            loss=loss.item(),
            acc=acc
        )
        bar.next()
    bar.finish()
    
    np.save('labels', np.asarray(obj_labels))
    np.save('preds', np.asarray(obj_preds))
    print('location loss is %.3f ' % (losses / count))

    return accs / count


def location_model_eva(model, data_loader):
    model.eval()
    accs = 0
    recalls = 0
    num = len(data_loader.dataset)
    count = 0
    bar = Bar('processing', max=len(data_loader))
    results = {}
    preds = {}
    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            audio_data, posi_img_data, nega_img_data, posi_label, nega_label, img_path, _ = data
            audio_data, image_data, av_labels, _ = batch_organize(audio_data, posi_img_data, nega_img_data, posi_label,
                                                                  nega_label)
            audio_data, image_data = audio_data.type(torch.FloatTensor).cuda(), \
                                     image_data.type(torch.FloatTensor).cuda()

            av_outputs, av_map, v_fea, _, v_cls, _ = model(image_data, audio_data)
            
#            v_fea = v_fea.view(v_fea.shape[0], 512, 196).permute(0, 2, 1).contiguous()
#            v_cam = model.v_class_layer(v_fea)
#            v_cam = v_cam.permute(0, 2, 1).view(v_fea.shape[0], 11, 14, 14)
#            v_cam = torch.sigmoid(v_cam)
#            v_cam_ = v_cam.detach().cpu().numpy()
#            v_cam = torch.nn.functional.interpolate(v_cam, (224, 224), mode='bilinear')

            accs += eva_metric(av_outputs.detach().cpu().numpy(), av_labels.numpy())[0]
            recalls += eva_metric(av_outputs.detach().cpu().numpy(), av_labels.numpy())[1]
            count += 1
            v_cls = torch.nn.functional.softmax(v_cls, 1)
            v_cls = v_cls.detach().cpu().numpy()
            av_map_ = av_map.detach().cpu().numpy()
            av_map = torch.nn.functional.interpolate(av_map, (224, 224), mode='bilinear')
            image_data = image_data.permute(0, 2, 3, 1)
            visualize(image_data, av_map, i)
            for k in range(len(img_path)):
                results[img_path[k][51:]] = av_map_[2*k]
                preds[img_path[k][51:]] = v_cls[2*k]
#                results[img_path[k][51:]] = v_cam_[2*k][corr[int(posi_label[k])]]
#                visualize_perimage(image_data[2*k], v_cam[2*k][corr[int(posi_label[k])]], i, k)
            bar.suffix  = '({batch}/{size}) Acc: {acc:.3f}'.format(
                batch=i + 1,
                size=len(data_loader),
                acc=eva_metric(av_outputs.detach().cpu().numpy(), av_labels.numpy())[0]
            )
            bar.next()
    bar.finish()
    pickle.dump(results, open('single.pkl', 'wb'))
    pickle.dump(preds, open('preds.pkl', 'wb'))

    return accs / count, recalls / count


def visualize_perimage(image, cam, e, i):
    image = image.cpu().numpy()
    cam = cam.detach().cpu().numpy()
    image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    image = np.clip(image, 0, 1)
    cam = cam * 255
    cam = cv2.applyColorMap(cam.astype(np.uint8), cv2.COLORMAP_JET)
    cam = cam[:, :, ::-1] / 255
    plt.imsave('vis/img_'+str(e)+'_'+str(i)+'.jpg', 0.5*cam+0.5*image)


def visualize(images, cams, e):
    images = images.cpu().numpy()
    cams = cams.detach().cpu().numpy()
    for i in range(images.shape[0]):
        if i % 2 == 0:
            cor = '_cor'
        else:
            cor = '_not'
        cam = cams[i, 0]
#        cam[cam>0.1] = 1
#        cam[cam<0.1] = 0
        image = images[i]
        image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        image = np.clip(image, 0, 1)
        cam = cam * 255
        cam = cv2.applyColorMap(cam.astype(np.uint8), cv2.COLORMAP_JET)
        cam = cam[:, :, ::-1] / 255
        plt.imsave('vis/img_'+str(e)+'_'+str(i)+cor+'.jpg', 0.5*cam+0.5*image)


def extract_feature(model, data_loader):
    print('extracting features...')
    model.eval()
    accs = 0
    num = len(data_loader.dataset)
    count = 0
    obj_features = []
    obj_labels = []
    img_dirs = []
    aud_dirs = []
    bar = Bar('processing', max=len(data_loader))
    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            audio_data, posi_img_data, nega_img_data, posi_label, nega_label, posi_dir, aud_dir = data
            audio_data, image_data, av_labels, class_label = batch_organize(audio_data, posi_img_data, nega_img_data,
                                                                            posi_label, nega_label)
            audio_data, image_data = audio_data.type(torch.FloatTensor).cuda(), \
                                     image_data.type(torch.FloatTensor).cuda()

            _, location_map, v_fea, _, _, _ = model(image_data, audio_data)

            location_map = location_map.detach().cpu().numpy()
            v_fea = v_fea.detach().cpu().numpy()

            for j in range(location_map.shape[0]):
                if av_labels[j].cpu().numpy() == 0:
                    continue
                obj_mask = np.uint8(location_map[j] > 0.05)
                obj_mask = location_dilation(obj_mask)

                img_rep = v_fea[j, :, :, :]  # such as batch x 512 x 4 x 14 x 14
                obj_rep = img_rep * obj_mask
                obj_rep = np.mean(obj_rep, (1, 2))

                obj_features.append(obj_rep)
                obj_labels.append(class_label[j].numpy())
                
            img_dirs.extend(list(posi_dir))
            aud_dirs.extend(list(aud_dir))
            if (i+1) % 100 == 0:
                np.save('obj_'+str(i+1).zfill(4), np.asarray(obj_features))
                obj_features = []
            bar.suffix  = '({batch}/{size})'.format(
                batch=i + 1,
                size=len(data_loader)
            )
            bar.next()
    bar.finish()
    
    obj_synthesis = []
    for i in range(1, 9):
        obj_synthesis.append(np.load('obj_'+str(100*i).zfill(4)+'.npy'))
    obj_synthesis.append(np.asarray(obj_features))

    return np.vstack(obj_synthesis), np.asarray(obj_labels), img_dirs, aud_dirs


def feature_clustering(data, label):
    data = normalize(data, norm='l2')
    labels = []
    for i in range(label.shape[0]):
        if label[i] not in labels:
            labels.append(label[i])
    count = len(labels)
    kmeans = cluster.KMeans(n_clusters=count, random_state=0).fit(data)
    cluster_label = kmeans.labels_
    score = metrics.normalized_mutual_info_score(label, cluster_label)
    pickle.dump(label, open('label.pkl', 'wb'))
    pickle.dump(cluster_label, open('cluster_label.pkl', 'wb'))

    return cluster_label, score


def class_model_train(model, data_loader, optimizer, criterion, args):
    model.v_class_layer = torch.nn.Linear(512, 11)
    model.v_class_layer.weight.data.normal_(0, 0.01)
    model.v_class_layer.bias.data.zero_()
    model.v_class_layer.cuda()
    
    model.a_class_layer = torch.nn.Linear(512, 11)
    model.a_class_layer.weight.data.normal_(0, 0.01)
    model.a_class_layer.bias.data.zero_()
    model.a_class_layer.cuda()
    model.train()

    params = list(model.v_class_layer.parameters()) + list(model.a_class_layer.parameters())
    optimizer_cl = torch.optim.Adam(params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0001)

    for j in range(args.class_iter):
        count = 0
        losses = 0
        for i, data in enumerate(data_loader, 0):
            # if i%200==0:
            #    print('class batch:%d' % i)
            aud, img, label = data
            img, aud, label = img.type(torch.FloatTensor).cuda(), \
                              aud.type(torch.FloatTensor).cuda(), \
                              label.type(torch.LongTensor).cuda()

            _, _, _, _, v_logits, a_logits  = model(img, aud)
            loss = criterion(v_logits, label) + criterion(a_logits, label)
            optimizer.zero_grad()
            optimizer_cl.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer_cl.step()

            losses += loss.detach().cpu().numpy()
            count += 1

        print('class loss is %.3f ' % (losses / count))


def main():
    parser = argparse.ArgumentParser(description='AID_PRETRAIN')
    parser.add_argument('--data_list_dir', type=str,
                        default='./data/data_indicator/music/solo')
    parser.add_argument('--data_dir', type=str, default='/media/yuxi/Data/MUSIC_Data/data/solo/')
    parser.add_argument('--mode', type=str, default='train', help='train/val/test')
    parser.add_argument('--json_file', type=str,default='./data/MUSIC_label/MUSIC_solo_videos.json')
    
    parser.add_argument('--use_class_task', type=int, default=1, help='whether to use class task')
    parser.add_argument('--init_num', type=int, default=0, help='epoch number for initializing the location model')
    parser.add_argument('--use_pretrain', type=int, default=0, help='whether to init from ckpt')
    parser.add_argument('--ckpt_file', type=str, default='location_net_009_0.665.pth', help='pretrained model name')
    parser.add_argument('--enable_img_augmentation', type=int, default=1, help='whether to augment input image')
    parser.add_argument('--enable_audio_augmentation', type=int, default=1, help='whether to augment input audio')
    parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='training batch size')
    parser.add_argument('--epoch', type=int, default=2000, help='training epoch')
    parser.add_argument('--class_iter', type=int, default=3, help='training iteration for classification model')
    parser.add_argument('--gpu_ids', type=str, default='[0,1,2,3]', help='USING GPU IDS e.g.\'[0,4]\'')
    parser.add_argument('--num_threads', type=int, default=4, help='number of threads')
    parser.add_argument('--seed', type=int, default=10)
    args = parser.parse_args()


    train_list_file = os.path.join(args.data_list_dir, 'solo_training_1.txt')
    val_list_file = os.path.join(args.data_list_dir, 'solo_validation.txt')
    test_list_file = os.path.join(args.data_list_dir, 'solo_testing.txt')

    train_dataset = MUSIC_Dataset(args.data_dir, train_list_file, args)
    val_dataset = MUSIC_Dataset(args.data_dir, val_list_file, args)
    test_dataset = MUSIC_Dataset(args.data_dir, test_list_file, args)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_threads)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_threads)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_threads)

    # net setup
    visual_backbone = resnet18(pretrained=True, modal='vision')
    audio_backbone = resnet18(modal='audio')
    av_model = Location_Net(visual_net=visual_backbone, audio_net=audio_backbone)

    if args.use_pretrain:
        PATH = args.ckpt_file
        state = torch.load(PATH)
        av_model.load_state_dict(state, strict=False)
    av_model_cuda = av_model.cuda()
    av_model_cuda.conv_av.weight.data.fill_(5.)

    loss_func_BCE_location = torch.nn.BCELoss(reduce=True)
    loss_func_BCE_class = torch.nn.CrossEntropyLoss()  # torch.nn.BCELoss(reduce=True)

    params = list(av_model_cuda.parameters())
    optimizer = optim.Adam(params=params[:-4], lr=args.learning_rate, betas=(0.9, 0.999),
                                    weight_decay=0.0001)

    init_num = 0
    for e in range(1, args.epoch):
        print('Epoch is %03d' % e)
        # av_model_cuda.class_layer = None
        # train_location_acc = location_model_train(av_model_cuda, train_dataloader, optimizer_location, loss_func_BCE_location)
        # eva_location_acc, eva_location_rec = location_model_eva(av_model_cuda, val_dataloader)

#        _, img_nmi_score = feature_clustering(img_features, labels)
        obj_nmi_score = 0.0
        img_nmi_score = 0.0
        
        train_location_acc = location_model_train(av_model_cuda, train_dataloader, optimizer,
                                                  loss_func_BCE_location)
        eva_location_acc, eva_location_rec = location_model_eva(av_model_cuda, val_dataloader)

        if init_num > args.init_num and args.use_class_task:
            # pseudo_label_ = np.zeros((pseudo_label.size, 11))           # here for multi-label classification
            # pseudo_label_[np.arange(pseudo_label.size), pseudo_label] = 1
            obj_features, labels, img_dirs, aud_dirs = extract_feature(av_model_cuda, train_dataloader)
            pseudo_label, obj_nmi_score = feature_clustering(obj_features, labels)
            obj_fea = []
            for i in range(11):
                cur_idx = pseudo_label == i
                cur_fea = obj_features[cur_idx]
                obj_fea.append(np.mean(cur_fea, 0))
            np.save('obj_feature_softmax_avg_fc.npy', obj_fea)
            
            class_dataset = MUSIC_AV_Classify(img_dirs, aud_dirs, pseudo_label, args)
            class_dataloader = DataLoader(dataset=class_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_threads)

            class_model_train(av_model_cuda, class_dataloader, optimizer, loss_func_BCE_class, args)

#        eva_location_acc, eva_location_rec = location_model_eva(av_model_cuda, test_dataloader)
#        print(eva_location_acc, eva_location_rec, obj_nmi_score)

        print('train acc is %.3f, val acc is %.3f, obj_NMI is %.3f, img_NMI is %.3f' % (train_location_acc, eva_location_acc, obj_nmi_score, img_nmi_score))
        init_num += 1
        if e % 1 == 0:
            PATH = 'ckpt/stage-one/location_cluster_net_softmax_%03d_%.3f_%.3f.pth' % (e, eva_location_acc, obj_nmi_score)
            torch.save(av_model_cuda.state_dict(), PATH)


if __name__ == '__main__':
    main()


