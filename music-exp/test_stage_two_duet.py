import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
from data.MUSIC_dataset import MUSIC_Dataset

from model.base_model2 import resnet18
from model.location_model import Location_Net_stage_two, Location_Net_stage_one

from sklearn import cluster, metrics
import numpy as np
from sklearn.preprocessing import normalize
import cv2
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


def returnCAM(feature_conv):
    cam = feature_conv
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    return cam


def visualize_model(model, av_model_fix, data_loader, obj_rep):
    
    corr = np.load('corr.npy')
    keylist = ['flute', 'acoustic_guitar', 'accordion', 'xylophone', 'erhu', 'tuba', 
               'saxophone', 'cello', 'violin', 'clarinet', 'trumpet']
    # keylist = pickle.load(open('keylist.pkl', 'rb'))
    duetkeys = {13: 'accordion', 3: 'acoustic_guitar', 1: 'cello', 12: 'flute', 8: 'saxophone', 
                10: 'trumpet', 11: 'violin'}
    
    model.eval()
    accs = 0
    num = len(data_loader.dataset)
    count = 0
    results = {}
    avmaps = {}
    obj_rep_cuda = torch.from_numpy(obj_rep).type(torch.FloatTensor).cuda()
    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            audio_data, posi_img_data, nega_img_data, posi_label, nega_label, path, _ = data
            audio_data, image_data, av_labels = batch_organize(audio_data, posi_img_data, nega_img_data)
            audio_data, image_data = audio_data.type(torch.FloatTensor).cuda(), \
                                     image_data.type(torch.FloatTensor).cuda()
            # print(image_data.shape, audio_data.shape, obj_rep_cuda.shape)
            av_outputs, av_maps, a_logits, sounding_objects, obj_mask = model(image_data, audio_data, obj_rep_cuda, minimize=True)

            with torch.no_grad():
                _, _, _, _, _, audio_class = av_model_fix(image_data, audio_data)
            
            # cams
            snd_localization = av_maps
            snd_localization = snd_localization.detach().cpu().numpy()
            snd_localization = snd_localization[::2]
            imgs = posi_img_data.cpu().numpy()


            for j in range(snd_localization.shape[0]):
                CAM = np.uint8(255 * snd_localization[j, 0, :, :])

                current_img = np.transpose(imgs[j, :, :, :], [1, 2, 0])
                current_img = current_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                current_img = np.clip(current_img, 0, 1)

                current_img = np.uint8(255 * current_img)
                height, width, _ = current_img.shape
                heatmap = cv2.applyColorMap(cv2.resize(CAM, (width, height)), cv2.COLORMAP_JET)

                current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2RGB)
                result = heatmap * 0.3 + current_img * 0.5

                # if not os.path.exists(os.path.join('cam/result/bayes2/', args.av_net_ckpt)):
                #    os.mkdir(os.path.join('cam/result/bayes2/', args.av_net_ckpt))
                file_name = '%04d_' % i + '%04d_sounding_area' % j + '.jpg'
                # cv2.imwrite(os.path.join('visualize/stage2/', file_name), result)


            # obj
            obj_mask = torch.softmax(obj_mask*3, 1)

            obj_localization = obj_mask
            obj_localization = obj_localization.detach().cpu().numpy()
            obj_localization = obj_localization[::2]
            imgs = posi_img_data.cpu().numpy()
            
            audio_class = audio_class[::2]
            audio_class = audio_class.detach().cpu().numpy()
            sounding_objects_idx = np.argsort(audio_class, axis=1)

            for j in range(obj_localization.shape[0]):
                results[path[j][-38:]] = obj_localization[j]
                avmaps[path[j][-38:]] = snd_localization[j]
                
                # for k in range(11):
                #     CAM = np.uint8(255 * obj_localization[j, k, :, :])
                #     #CAM = np.uint8(255 * obj_localization[j, sounding_objects_idx[j, -1-idx], :, :])

                #     current_img = np.transpose(imgs[j, :, :, :], [1, 2, 0])
                #     current_img = current_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                #     current_img = np.clip(current_img, 0, 1)

                #     current_img = np.uint8(255 * current_img)
                #     for b in box:
                #         current_img = cv2.rectangle(current_img, (int(b[0]), int(b[1])), (int(b[0]+b[2]), int(b[1]+b[3])), (0, 255, 0), 2)
                #     height, width, _ = current_img.shape
                #     heatmap = cv2.applyColorMap(cv2.resize(CAM, (width, height)), cv2.COLORMAP_JET)

                #     current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2RGB)
                #     result = heatmap * 0.3 + current_img * 0.5

                #     # if not os.path.exists(os.path.join('cam/result/bayes2/', args.av_net_ckpt)):
                #     #    os.mkdir(os.path.join('cam/result/bayes2/', args.av_net_ckpt))
                #     file_name = '%04d_' % i + '%04d_' % j +  '%04d_object' % k + '.jpg'
                #     cv2.imwrite(os.path.join('visualize/stage2/', file_name), result)
                
                # for box in boxs[j]:
                #     category = int(box[-1])
                #     index = corr[keylist.index(duetkeys[category])]
                #     index = int(index)
                #     box = np.array(box[:4]) * 224
                    
                #     CAM = np.uint8(255 * obj_localization[j, index, :, :])
                #     #CAM = np.uint8(255 * obj_localization[j, sounding_objects_idx[j, -1-idx], :, :])

                #     current_img = np.transpose(imgs[j, :, :, :], [1, 2, 0])
                #     current_img = current_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                #     current_img = np.clip(current_img, 0, 1)

                #     current_img = np.uint8(255 * current_img)
                #     current_img = cv2.rectangle(current_img, (int(box[0]), int(box[1])), (int(box[0]+box[2]), int(box[1]+box[3])), (0, 255, 0), 2)
                #     height, width, _ = current_img.shape
                #     heatmap = cv2.applyColorMap(cv2.resize(CAM, (width, height)), cv2.COLORMAP_JET)

                #     current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2RGB)
                #     result = heatmap * 0.3 + current_img * 0.5

                #     # if not os.path.exists(os.path.join('cam/result/bayes2/', args.av_net_ckpt)):
                #     #    os.mkdir(os.path.join('cam/result/bayes2/', args.av_net_ckpt))
                #     file_name = '%04d_' % i + '%04d_' % j +  '%04d_object' % index + '.jpg'
                #     cv2.imwrite(os.path.join('visualize/stage2/', file_name), result)
    
    pickle.dump(results, open('syn_objs.pkl', 'wb'))
    pickle.dump(avmaps, open('syn_avmaps.pkl', 'wb'))



def main():
    parser = argparse.ArgumentParser(description='AID_PRETRAIN')
    parser.add_argument('--data_list_dir', type=str,
                        default='./data/data_indicator/music/duet')
    parser.add_argument('--data_dir', type=str, default='/home/ruiq/Music/duet/duet/')
    parser.add_argument('--mode', type=str, default='test', help='train/val/test')
    parser.add_argument('--json_file', type=str,default='./data/MUSIC_label/MUSIC_duet_videos.json')
    
    parser.add_argument('--use_class_task', type=int, default=1, help='whether to use class task')
    parser.add_argument('--init_num', type=int, default=0, help='epoch number for initializing the location model')
    parser.add_argument('--use_pretrain', type=int, default=1, help='whether to init from ckpt')
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
    parser.add_argument('--cluster', type=int, default=11)
    parser.add_argument('--mask', type=float, default=0.05)
    args = parser.parse_args()

    val_list_file = os.path.join(args.data_list_dir, 'duet_pairs_val.txt')
    val_dataset = MUSIC_Dataset(args.data_dir, val_list_file, args)

    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_threads)


    # net setup
    visual_backbone = resnet18(modal='vision', pretrained=True)
    audio_backbone = resnet18(modal='audio')
    av_model = Location_Net_stage_two(visual_net=visual_backbone, audio_net=audio_backbone, cluster=args.cluster)
    if args.use_pretrain:
        PATH = args.ckpt_file
        state = torch.load(PATH)
        av_model.load_state_dict(state)
        print(PATH)
    av_model_cuda = av_model.cuda()

    # fixed model
    visual_backbone_fix = resnet18(modal='vision', pretrained=True)
    audio_backbone_fix = resnet18(modal='audio')
    av_model_fix = Location_Net_stage_one(visual_net=visual_backbone_fix, audio_net=audio_backbone_fix, cluster=args.cluster)
    if args.use_pretrain:
        PATH = os.path.join('ckpt/stage_one_%.2f_%d/' % (args.mask, args.cluster), 'location_cluster_net_iter_010_av_class.pth')
        state = torch.load(PATH)
        av_model_fix.load_state_dict(state)
        print('loaded weights')
    av_model_fix_cuda = av_model_fix.cuda()



    obj_rep = np.load('obj_features_%.2f_%d/obj_feature_softmax_avg_fc_epoch_10_av_entire.npy' % (args.mask, args.cluster))

    eva_location_acc = visualize_model(av_model_cuda, av_model_fix_cuda, val_dataloader, obj_rep)


if __name__ == '__main__':
    main()

