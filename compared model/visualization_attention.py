import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
from data.MUSIC_dataset import MUSIC_Dataset

from model.base_model2 import resnet18
from model.attention_net import Attention_Net

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


def location_model_train(model, data_loader, optimizer, criterion):
    model.train()
    accs = 0
    count = 0
    losses = 0
    for i, data in enumerate(data_loader, 0):
        if i % 200 == 0:
            print('location batch:%d' % i)
        audio_data, posi_img_data, nega_img_data = data
        audio_data, image_data, av_labels = batch_organize(audio_data, posi_img_data, nega_img_data)

        audio_data, image_data, av_labels = audio_data.type(torch.FloatTensor).cuda(), \
                                            image_data.type(torch.FloatTensor).cuda(), \
                                            av_labels.type(torch.FloatTensor).cuda()

        optimizer.zero_grad()
        av_outputs, av_maps = model(image_data, audio_data)
        loss = criterion(av_outputs, av_labels)
        loss.backward()
        optimizer.step()

        losses += loss.detach().cpu().numpy()
        acc = eva_metric(av_outputs.detach().cpu().numpy(), av_labels.cpu().numpy())
        accs += acc
        count += 1

    print('location loss is %.3f ' % (losses / count))

    return accs / count


def returnCAM(feature_conv):
    size_upsample = (256, 256)

    cam = feature_conv
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam+0.000001)

    cam_img = np.uint8(255 * cam_img)
    # output_cam.append(cv2.resize(cam_img, size_upsample))


    return cam_img


def location_model_eva(model, data_loader):
    model.eval()
    accs = 0
    num = len(data_loader.dataset)
    count = 0

    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            audio_data, posi_img_data, nega_img_data, _, _, _, _ = data
            audio_data, image_data, av_labels = batch_organize(audio_data, posi_img_data, nega_img_data)
            audio_data, image_data = audio_data.type(torch.FloatTensor).cuda(), image_data.type(
                torch.FloatTensor).cuda()

            _, av_atten, _ = model(image_data, audio_data)

            obj_localization = av_atten.detach().cpu().numpy()
            obj_localization = obj_localization[::2]
            imgs = posi_img_data.cpu().numpy()

            for j in range(obj_localization.shape[0]):
                map = np.reshape(obj_localization[j, :, 0], (14, 14))
                CAM = returnCAM(map)

                current_img = np.transpose(imgs[j, :, :, :], [1, 2, 0])
                current_img = np.uint8(255 * current_img)
                height, width, _ = current_img.shape
                heatmap = cv2.applyColorMap(cv2.resize(CAM, (width, height)), cv2.COLORMAP_JET)

                current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2RGB)
                result = heatmap * 0.3 + current_img * 0.5

                # if not os.path.exists(os.path.join('cam/result/bayes2/', args.av_net_ckpt)):
                #    os.mkdir(os.path.join('cam/result/bayes2/', args.av_net_ckpt))
                file_name = '%04d_' % i + '%04d_0' % j + '.jpg'
                cv2.imwrite(os.path.join('visualize/att/', file_name), result)



def main():
    parser = argparse.ArgumentParser(description='AID_PRETRAIN')
    parser.add_argument('--data_list_dir', type=str,
                        default='/mnt/home/hudi/location_sound/data/data_indicator/music/solo')
    parser.add_argument('--data_dir', type=str, default='/mnt/scratch/hudi/MUSIC/solo/')
    parser.add_argument('--mode', type=str, default='test', help='train/val/test')
    parser.add_argument('--json_file', type=str,
                        default='/mnt/home/hudi/location_sound/data/MUSIC_label/MUSIC_solo_videos.json')

    parser.add_argument('--use_class_task', type=int, default=1, help='whether to use class task')
    parser.add_argument('--init_num', type=int, default=0, help='epoch number for initializing the location model')
    parser.add_argument('--use_pretrain', type=int, default=1, help='whether to init from ckpt')
    parser.add_argument('--ckpt_file', type=str, default='att_stage_one_024_0.812_rand.pth',
                        help='pretrained model name')
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

    if args.init_num != 0 and args.use_pretrain:
        import sys
        print('If use ckpt, do not recommend to set init_num to 0.')
        sys.exit()

    val_list_file = os.path.join(args.data_list_dir, 'solo_pairs_val.txt')
    test_list_file = os.path.join(args.data_list_dir, 'solo_pairs_test.txt')

    val_dataset = MUSIC_Dataset(args.data_dir, val_list_file, args)
    test_dataset = MUSIC_Dataset(args.data_dir, test_list_file, args)

    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_threads)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_threads)

    # net setup
    visual_backbone = resnet18(modal='vision')
    audio_backbone = resnet18(modal='audio')
    av_model = Attention_Net(visual_net=visual_backbone, audio_net=audio_backbone)

    if args.use_pretrain:
        PATH = os.path.join('ckpt/att/', args.ckpt_file)
        state = torch.load(PATH)
        av_model.load_state_dict(state)
        print(PATH)
    av_model_cuda = av_model.cuda()


    location_model_eva(av_model_cuda, val_dataloader)

if __name__ == '__main__':
    main()





