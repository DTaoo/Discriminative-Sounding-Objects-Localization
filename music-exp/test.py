import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
from data.syn_dataset import MUSIC_Dataset

from model.base_model2 import resnet18
from model.location_model import Location_Net_stage_two, Location_Net_stage_one

from sklearn import cluster, metrics
import numpy as np
from sklearn.preprocessing import normalize
import cv2

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
    model.eval()
    accs = 0
    num = len(data_loader.dataset)
    count = 0
    obj_rep_cuda = torch.from_numpy(obj_rep).type(torch.FloatTensor).cuda()
    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            audio_data, posi_img_data, nega_img_data = data
            audio_data, image_data, av_labels = batch_organize(audio_data, posi_img_data, nega_img_data)
            audio_data, image_data = audio_data.type(torch.FloatTensor).cuda(), image_data.type(torch.FloatTensor).cuda()

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
                cv2.imwrite(os.path.join('visualize/stage2/', file_name), result)


            # obj
            obj_mask = torch.softmax(obj_mask, 1)

            obj_localization = obj_mask
            obj_localization = obj_localization.detach().cpu().numpy()
            obj_localization = obj_localization[::2]
            imgs = posi_img_data.cpu().numpy()

            audio_class = audio_class[::2]
            audio_class = audio_class.detach().cpu().numpy()
            sounding_objects_idx = np.argsort(audio_class, axis=1)

            for j in range(obj_localization.shape[0]):
                for idx in range(11):
                    CAM = np.uint8(255 * obj_localization[j, sounding_objects_idx[j, -1-idx], :, :])

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
                    file_name = '%04d_' % i + '%04d_' % j +  '%04d_object' % sounding_objects_idx[j, -1-idx] + '.jpg'
                    cv2.imwrite(os.path.join('visualize/stage2/', file_name), result)




def main():
    parser = argparse.ArgumentParser(description='AID_PRETRAIN')
    parser.add_argument('--data_list_dir', type=str,
                        default='./data/data_indicator/music/solo')
    parser.add_argument('--data_dir', type=str, default='/MUSIC/solo/')
    parser.add_argument('--mode', type=str, default='train', help='train/val/test')
    parser.add_argument('--json_file', type=str,
                        default='./data/MUSIC_label/MUSIC_solo_videos.json')

    parser.add_argument('--use_class_task', type=int, default=1, help='whether to use class task')
    parser.add_argument('--init_num', type=int, default=8, help='epoch number for initializing the location model')
    parser.add_argument('--use_pretrain', type=int, default=1, help='whether to init from ckpt')
    parser.add_argument('--ckpt_file', type=str, default='location_cluster_net_003_0.882_avg_whole.pth', help='pretrained model name')
    parser.add_argument('--enable_img_augmentation', type=int, default=1, help='whether to augment input image')
    parser.add_argument('--enable_audio_augmentation', type=int, default=1, help='whether to augment input audio')
    parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='training batch size')
    parser.add_argument('--epoch', type=int, default=2000, help='training epoch')
    parser.add_argument('--class_iter', type=int, default=3, help='training iteration for classification model')
    parser.add_argument('--gpu_ids', type=str, default='[0,1,2,3]', help='USING GPU IDS e.g.\'[0,4]\'')
    parser.add_argument('--num_threads', type=int, default=12, help='number of threads')
    parser.add_argument('--seed', type=int, default=10)
    args = parser.parse_args()

    train_dataset = MUSIC_Dataset(args)
    args_test = args
    args_test.mode = 'test'
    val_dataset = MUSIC_Dataset(args_test)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_threads)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_threads)


    # net setup
    visual_backbone = resnet18(modal='vision', pretrained=True)
    audio_backbone = resnet18(modal='audio')
    av_model = Location_Net_stage_two(visual_net=visual_backbone, audio_net=audio_backbone)
    if args.use_pretrain:
        PATH = os.path.join('ckpt/stage_two_cosine2/', args.ckpt_file)
        state = torch.load(PATH)
        av_model.load_state_dict(state)
        print(PATH)
    av_model_cuda = av_model.cuda()


    # fixed model
    visual_backbone_fix = resnet18(modal='vision', pretrained=True)
    audio_backbone_fix = resnet18(modal='audio')
    av_model_fix = Location_Net_stage_one(visual_net=visual_backbone_fix, audio_net=audio_backbone_fix)
    if args.use_pretrain:
        PATH = os.path.join('ckpt/stage_one_cosine2/', 'location_cluster_net_iter_006_av_class.pth')
        state = torch.load(PATH)
        av_model_fix.load_state_dict(state)
        print('loaded weights')
    av_model_fix_cuda = av_model_fix.cuda()



    obj_rep = np.load('obj_features2/obj_feature_softmax_avg_fc_epoch_6_av_entire.npy')

    eva_location_acc = visualize_model(av_model_cuda, av_model_fix_cuda, val_dataloader, obj_rep)


if __name__ == '__main__':
    main()

