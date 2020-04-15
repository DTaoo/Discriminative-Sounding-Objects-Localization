import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
from data.MUSIC_dataset import MUSIC_Dataset, MUSIC_AV_Classify

from model.base_model2 import resnet18
from model.dmc_model import DMC_NET

from sklearn import cluster, metrics
import numpy as np
from sklearn.preprocessing import normalize
from torch import nn
import torch.nn.functional as F

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
    return correct / predict.shape[0]

def eva_metric2(predict, gt, pair_num=2):

    num = int(predict.shape[0]/pair_num)
    correct = 0
    for i in range(num):
        pos = predict[pair_num*i]
        flag = True
        for j in range(pair_num-1):
            neg = predict[pair_num*i+j+1]
            if pos >= neg:
                flag = False
        if flag == True:
            correct += 1

    return correct / num

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin=5.):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output, target, size_average=True):
        distances = output.pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


def location_model_train(model, data_loader, optimizer, criterion):
    model.train()
    accs = 0
    count = 0
    losses = 0
    for i, data in enumerate(data_loader, 0):
        if i % 200 == 0:
            print('location batch:%d' % i)
        audio_data, posi_img_data, nega_img_data, posi_label, nega_label, _, _ = data
        audio_data, image_data, av_labels, class_labels = batch_organize(audio_data, posi_img_data, nega_img_data, posi_label, nega_label)

        audio_data, image_data, av_labels = audio_data.type(torch.FloatTensor).cuda(), \
                                            image_data.type(torch.FloatTensor).cuda(), \
                                            av_labels.type(torch.FloatTensor).cuda()

        optimizer.zero_grad()
        av_outputs, _, _ = model(image_data, audio_data)
        loss = criterion(av_outputs, av_labels)
        loss.backward()
        optimizer.step()

        losses += loss.detach().cpu().numpy()
        acc = eva_metric2(av_outputs.detach().cpu().numpy(), av_labels.cpu().numpy())
        accs += acc
        count += 1

    print('location loss is %.3f ' % (losses / count))

    return accs / count


def location_model_eva(model, data_loader):
    model.eval()
    accs = 0
    num = len(data_loader.dataset)
    count = 0

    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            audio_data, posi_img_data, nega_img_data, posi_label, nega_label, _, _ = data
            audio_data, image_data, av_labels, class_labels = batch_organize(audio_data, posi_img_data, nega_img_data,
                                                                             posi_label, nega_label)

            audio_data, image_data = audio_data.type(torch.FloatTensor).cuda(), image_data.type(torch.FloatTensor).cuda()

            av_outputs, _, _ = model(image_data, audio_data)

            accs += eva_metric2(av_outputs.detach().cpu().numpy(), av_labels.numpy())
            count += 1

    return accs / count



def main():
    parser = argparse.ArgumentParser(description='AID_PRETRAIN')
    parser.add_argument('--data_list_dir', type=str,
                        default='/mnt/home/hudi/location_sound/data/data_indicator/music/solo')
    parser.add_argument('--data_dir', type=str, default='/mnt/scratch/hudi/MUSIC/solo/')
    parser.add_argument('--mode', type=str, default='train', help='train/val/test')
    parser.add_argument('--json_file', type=str,
                        default='/mnt/home/hudi/location_sound/data/MUSIC_label/MUSIC_solo_videos.json')

    parser.add_argument('--use_class_task', type=int, default=1, help='whether to use class task')
    parser.add_argument('--init_num', type=int, default=0, help='epoch number for initializing the location model')
    parser.add_argument('--use_pretrain', type=int, default=0, help='whether to init from ckpt')
    parser.add_argument('--ckpt_file', type=str, default='location_cluster_net_softmax_009_0.709.pth', help='pretrained model name')
    parser.add_argument('--enable_img_augmentation', type=int, default=1, help='whether to augment input image')
    parser.add_argument('--enable_audio_augmentation', type=int, default=1, help='whether to augment input audio')
    parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='training batch size')
    parser.add_argument('--epoch', type=int, default=2000, help='training epoch')
    parser.add_argument('--class_iter', type=int, default=30, help='training iteration for classification model')
    parser.add_argument('--gpu_ids', type=str, default='[0,1,2,3]', help='USING GPU IDS e.g.\'[0,4]\'')
    parser.add_argument('--num_threads', type=int, default=12, help='number of threads')
    parser.add_argument('--seed', type=int, default=10)
    args = parser.parse_args()


    train_list_file = os.path.join(args.data_list_dir, 'solo_pairs_train.txt')
    val_list_file = os.path.join(args.data_list_dir, 'solo_pairs_val.txt')
    test_list_file = os.path.join(args.data_list_dir, 'solo_pairs_test.txt')

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
    visual_backbone = resnet18(modal='vision',pretrained=True)
    audio_backbone = resnet18(modal='audio')
    av_model = DMC_NET(visual_net=visual_backbone, audio_net=audio_backbone)

    if args.use_pretrain:
        PATH = os.path.join('ckpt/stage_one/', args.ckpt_file)
        state = torch.load(PATH)
        av_model.load_state_dict(state, strict=False)
    av_model_cuda = av_model.cuda()

    loss_func = ContrastiveLoss()

    optimizer = optim.Adam(params=av_model_cuda.parameters(), lr=args.learning_rate, betas=(0.9, 0.999),
                           weight_decay=0.0001)

    init_num = 0
    for e in range(args.epoch):
        print('Epoch is %03d' % e)

        train_location_acc = location_model_train(av_model_cuda, train_dataloader, optimizer, loss_func)
        eva_location_acc   = location_model_eva(av_model_cuda, val_dataloader)

        print('train acc is %.3f, val acc is %.3f' % (train_location_acc, eva_location_acc))
        init_num += 1
        if e % 3 == 0:
            PATH = 'ckpt/dmc/dmc_stage_one_%03d_%.3f.pth' % (e, eva_location_acc)
            torch.save(av_model_cuda.state_dict(), PATH)


if __name__ == '__main__':
    main()





