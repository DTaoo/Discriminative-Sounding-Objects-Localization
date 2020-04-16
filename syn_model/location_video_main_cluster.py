import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
from data.MUSIC_dataset import MUSIC_Dataset, MUSIC_AV_Classify
import cv2
import matplotlib.pyplot as plt

from model.base_model import resnet18
from model.location_model import Location_Net_stage_two

from sklearn import cluster, metrics
import numpy as np
from sklearn.preprocessing import normalize
import pickle

def batch_organize(audio_data, posi_img_data, nega_img_data):
    batch_audio_data = torch.zeros(audio_data.shape[0] * 2, audio_data.shape[1], audio_data.shape[2],
                                   audio_data.shape[3])
    batch_image_data = torch.zeros(posi_img_data.shape[0] * 2, posi_img_data.shape[1], posi_img_data.shape[2],
                                   posi_img_data.shape[3])
    batch_labels = torch.zeros(audio_data.shape[0] * 2)
    for i in range(audio_data.shape[0]):
        batch_audio_data[i * 2, :] = audio_data[i, :]
        batch_audio_data[i * 2 + 1, :] = audio_data[i, :]
        batch_image_data[i * 2, :] = posi_img_data[i, :]
        batch_image_data[i * 2 + 1, :] = nega_img_data[i, :]
        batch_labels[i * 2] = 1
        batch_labels[i * 2 + 1] = 0
    return batch_audio_data, batch_image_data, batch_labels


def eva_metric(predict, gt):
    correct = (np.round(predict) == gt).sum()
    return correct / predict.shape[0]


def location_model_train(model, av_model_fix, data_loader, optimizer, criterion_location, criterion_class, epoch, obj_rep):
    model.train()
    av_model_fix.eval()
    accs = 0
    count = 0
    losses_g = 0
    losses_l = 0
    obj_rep_cuda = torch.from_numpy(obj_rep).type(torch.FloatTensor).cuda()
    print(len(data_loader))
    for i, data in enumerate(data_loader, 0):
        if i % 200 == 0:
            print('location batch:%d' % i)
        audio_data, posi_img_data, nega_img_data, _,_,_,_ = data
        audio_data, image_data, av_labels = batch_organize(audio_data, posi_img_data, nega_img_data)

        audio_data, image_data, av_labels = audio_data.type(torch.FloatTensor).cuda(), \
                                            image_data.type(torch.FloatTensor).cuda(), \
                                            av_labels.type(torch.FloatTensor).cuda()

        model.zero_grad()
        av_outputs, _, _, sounding_objects, sounding_object_local = model(image_data, audio_data, obj_rep_cuda, epoch>10)

#        posi_outputs = av_outputs[::2].contiguous()
#        posi_outputs = torch.max(posi_outputs, 1)[0]
        posi_labels = av_labels[::2].contiguous()
#        nega_outputs = av_outputs[1::2].contiguous()
#        nega_outputs = nega_outputs.view(-1)
        posi_outputs, nega_outputs = av_outputs
        nega_labels = av_labels[1::2].contiguous()
        nega_labels = nega_labels.view(nega_labels.shape[0], 1).repeat([1, 4])
        nega_labels = nega_labels.view(-1)
#        print(posi_outputs.shape, posi_labels.shape, nega_outputs.shape, nega_labels.shape)
#        loss_global = criterion_location(av_outputs, av_labels)
        loss_global = criterion_location(posi_outputs, posi_labels) + criterion_location(nega_outputs, nega_labels)
        loss_global = 0.5 * loss_global

        with torch.no_grad():
            _, _, audio_class, _, _ = av_model_fix(image_data, audio_data, obj_rep_cuda)

        audio_class = audio_class[::2]
        sounding_objects = sounding_objects[::2]

        pseudo_label = torch.zeros_like(audio_class)
        idx = torch.argsort(audio_class, dim=1)
        pseudo_label[np.arange(audio_class.shape[0]), idx[:, -2]] = 1
        pseudo_label[np.arange(audio_class.shape[0]), idx[:, -1]] = 1
        pseudo_label = pseudo_label.cuda()

        #sounding_objects = torch.nn.functional.log_softmax(sounding_objects/5, dim=1)
        audio_class = torch.nn.functional.softmax(audio_class.detach(), dim=1)
        loss_local = criterion_class(sounding_objects, audio_class)

        if epoch <= 10:
            losses = loss_local
            losses.backward()
            optimizer.step()
        else:
            losses = loss_local + loss_global
            losses.backward()
            optimizer.step()

        losses_g += loss_global.detach().cpu().numpy()
        losses_l += loss_local.detach().cpu().numpy()
        
        av_outputs = torch.cat([posi_outputs, nega_outputs], 0)
        av_labels = torch.cat([posi_labels, nega_labels], 0)

        acc = eva_metric(av_outputs.detach().cpu().numpy(), av_labels.cpu().numpy())
        accs += acc
        count += 1

    print('Global matching loss is %.3f, local matching loss is %.3f ' % ((losses_g/count), (losses_l/count)))

    return accs / count


def location_model_eval(model, av_model_fix, data_loader, optimizer, criterion_location, criterion_class, epoch, obj_rep):
    model.eval()
    av_model_fix.eval()
    accs = 0
    count = 0
    losses_g = 0
    losses_l = 0
    corr = np.load('corr.npy')
    preds = {}
    obj_rep_cuda = torch.from_numpy(obj_rep).type(torch.FloatTensor).cuda()
    for i, data in enumerate(data_loader, 0):

        audio_data, posi_img_data, nega_img_data, posi_label, nega_label, img_path, boxs= data
        audio_data, image_data, av_labels = batch_organize(audio_data, posi_img_data, nega_img_data)

        audio_data, image_data, av_labels = audio_data.type(torch.FloatTensor).cuda(), \
                                            image_data.type(torch.FloatTensor).cuda(), \
                                            av_labels.type(torch.FloatTensor).cuda()

        model.zero_grad()
        av_outputs, av_maps, audio_class, sounding_objects, sounding_object_local = model(image_data, audio_data, obj_rep_cuda)
        
#        posi_outputs = av_outputs[::2].contiguous()
#        posi_outputs = torch.max(posi_outputs, 1)[0]
        posi_labels = av_labels[::2].contiguous()
#        nega_outputs = av_outputs[1::2].contiguous()
#        nega_outputs = nega_outputs.view(-1)
        posi_outputs, nega_outputs = av_outputs
        nega_labels = av_labels[1::2].contiguous()
        nega_labels = nega_labels.view(nega_labels.shape[0], 1).repeat([1, 4])
        nega_labels = nega_labels.view(-1)
#        print(posi_outputs.shape, posi_labels.shape, nega_outputs.shape, nega_labels.shape)
#        loss_global = criterion_location(av_outputs, av_labels)
        loss_global = criterion_location(posi_outputs, posi_labels) + criterion_location(nega_outputs, nega_labels)
        loss_global = 0.5 * loss_global

#        with torch.no_grad():
#            _, _, audio_class, _, _ = av_model_fix(image_data, audio_data, obj_rep_cuda)

        audio_class = audio_class[::2]
        sounding_objects = sounding_objects[::2]

        pseudo_label = torch.zeros_like(audio_class)
        idx = torch.argsort(audio_class, dim=1)
        pseudo_label[np.arange(audio_class.shape[0]), idx[:, -2]] = 1
        pseudo_label[np.arange(audio_class.shape[0]), idx[:, -1]] = 1
        pseudo_label = pseudo_label.cuda()

        #sounding_objects = torch.nn.functional.log_softmax(sounding_objects/5, dim=1)
        audio_class = torch.nn.functional.softmax(audio_class.detach()/0.7, dim=1)

        loss_local = criterion_class(sounding_objects, audio_class)

        losses_g += loss_global.detach().cpu().numpy()
        losses_l += loss_local.detach().cpu().numpy()

#        for k in range(len(img_path)):
#            preds[img_path[k][51:]] = sounding_object_local[2*k].detach().cpu().numpy()
#            preds[img_path[k][51:]] = av_maps[2*k].detach().cpu().numpy()
            
        av_maps = torch.nn.functional.interpolate(av_maps, (224, 224), mode='bilinear')
        sounding_object_local = torch.nn.functional.interpolate(sounding_object_local, (224, 224), mode='bilinear')
        image_data = image_data.permute(0, 2, 3, 1)
        visualize(image_data, av_maps, boxs, i)
#        visualize_object(image_data, sounding_object_local, boxs, i)
#        visualize_cam(image_data, sounding_object_local, posi_label, corr, i)
        av_outputs = torch.cat([posi_outputs, nega_outputs], 0)
        av_labels = torch.cat([posi_labels, nega_labels], 0)

        acc = eva_metric(av_outputs.detach().cpu().numpy(), av_labels.cpu().numpy())
        accs += acc
        count += 1

    print('Global matching loss is %.3f, local matching loss is %.3f ' % ((losses_g/count), (losses_l/count)))
    print('AVmap accuracy is %.3f' % (accs/count))
    pickle.dump(preds, open('duet.pkl', 'wb'))
    return accs / count


def visualize(images, cams, boxs, e):
    images = images.cpu().numpy()
    cams = cams.detach().cpu().numpy()
    for i in range(images.shape[0]):
        if i % 2 == 0:
            cor = '_cor'
        else:
            cor = '_not'
        cam = cams[i, 0]
        image = images[i]
        image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        image = np.clip(image, 0, 1)
        cam = cam * 255
        cam = cv2.applyColorMap(cam.astype(np.uint8), cv2.COLORMAP_JET)
        if i % 2 == 0:
            for box in boxs[i//2]:
                box = box * 224
                cv2.rectangle(cam, (int(box[0]), int(box[1])), (int(box[0]+box[2]), 
                                        int(box[1]+box[3])), (0, 255, 0), 2)
        cam = cam[:, :, ::-1] / 255
        plt.imsave('vis/img_'+str(e)+'_'+str(i)+cor+'.jpg', 0.5*cam+0.5*image)


def visualize_cam(images, cams, scores, corr, e):
    images = images.cpu().numpy()
    cams = cams.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()
    for i in range(images.shape[0]):
        if i % 2 != 0:
            continue
        score = scores[i//2]
        idx = np.argsort(score)[::-1]
        if corr is not None:
            idx = corr[idx]
        for k in range(2):
            cam = cams[i, int(idx[k])]
            image = images[i]
            image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            image = np.clip(image, 0, 1)
            cam = cam * 255
            cam = cv2.applyColorMap(cam.astype(np.uint8), cv2.COLORMAP_JET)
            cam = cam[:, :, ::-1] / 255
            plt.imsave('vis/img_'+str(e)+'_'+str(i)+'_'+str(idx[k])+'.jpg', 0.5*cam+0.5*image)


def visualize_object(images, cams, ks, e):
    images = images.cpu().numpy()
    cams = cams.detach().cpu().numpy()
    for i in range(images.shape[0]):
        if i % 2 != 0:
            continue
        for k in ks[i//2]:
            cam = cams[i, int(k[-1])]
            image = images[i]
            image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            image = np.clip(image, 0, 1)
            cam = cam * 255
            cam = cv2.applyColorMap(cam.astype(np.uint8), cv2.COLORMAP_JET)
            box = k[:4] * 224
            cv2.rectangle(cam, (int(box[0]), int(box[1])), (int(box[0]+box[2]), int(box[1]+box[3])), (0, 255, 0), 2)
            cam = cam[:, :, ::-1] / 255
            plt.imsave('vis/img_'+str(e)+'_'+str(i)+'_'+str(int(k[-1]))+'.jpg', 0.5*cam+0.5*image)



def main():
    parser = argparse.ArgumentParser(description='AID_PRETRAIN')
    parser.add_argument('--data_list_dir', type=str,
                        default='./data/data_indicator/music/duet')
    parser.add_argument('--data_dir', type=str, default='/media/yuxi/Data/MUSIC_Data/data/duet/')
    parser.add_argument('--mode', type=str, default='train', help='train/val/test')
    parser.add_argument('--json_file', type=str,default='./data/MUSIC_label/MUSIC_duet_videos.json')
    
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

    if args.init_num != 0 and args.use_pretrain:
        import sys
        print('If use ckpt, do not recommend to set init_num to 0.')
        sys.exit()

    train_list_file = os.path.join(args.data_list_dir, 'duet_pairs_train.txt')
    val_list_file = os.path.join(args.data_list_dir, 'duet_pairs_val.txt')

    train_dataset = MUSIC_Dataset(args.data_dir, train_list_file, args)
    val_dataset = MUSIC_Dataset(args.data_dir, val_list_file, args)


    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_threads)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_threads)

    # net setup
    visual_backbone = resnet18(modal='vision')
    audio_backbone = resnet18(modal='audio')
    av_model = Location_Net_stage_two(visual_net=visual_backbone, audio_net=audio_backbone)

    if args.use_pretrain:
        PATH = os.path.join(args.ckpt_file)
        state = torch.load(PATH)
        av_model.load_state_dict(state, strict=False)
    av_model_cuda = av_model.cuda()

    # fixed model
    visual_backbone_fix = resnet18(modal='vision')
    audio_backbone_fix = resnet18(modal='audio')
    av_model_fix = Location_Net_stage_two(visual_net=visual_backbone_fix, audio_net=audio_backbone_fix)

    if args.use_pretrain:
#        PATH = os.path.join(args.ckpt_file)
        PATH = 'ckpt/video_cluster/location_cluster_net_softmax_000_0.602_0.787.pth'
        state = torch.load(PATH)
        av_model_fix.load_state_dict(state, strict=False)
        print('loaded weights')
    av_model_fix_cuda = av_model_fix.cuda()

    # loading the object representation for stage_one
    obj_rep = np.load('obj_feature_softmax_avg_fc.npy')

    loss_func_BCE_location = torch.nn.BCELoss(reduce=True)
    loss_func_BCE_class = torch.nn.BCELoss(reduce=True)

    optimizer = optim.Adam(params=av_model_cuda.parameters(), lr=args.learning_rate, betas=(0.9, 0.999),
                           weight_decay=0.0001)

    init_num = 0
    for e in range(0, args.epoch):
        print('Epoch is %03d' % e)

        train_location_acc = location_model_train(av_model_cuda, av_model_fix_cuda, train_dataloader, optimizer,
                                                  loss_func_BCE_location, loss_func_BCE_class, e, obj_rep)
#        eval_location_acc = location_model_eval(av_model_cuda, av_model_fix_cuda, val_dataloader, optimizer,
#                                                  loss_func_BCE_location, loss_func_BCE_class, e, obj_rep)

        if e % 1 == 0:
            PATH = 'ckpt/synav/audioset_pretrain_%03d_%.3f.pth' % (e, train_location_acc)
            torch.save(av_model_cuda.state_dict(), PATH)

if __name__ == '__main__':
    main()
