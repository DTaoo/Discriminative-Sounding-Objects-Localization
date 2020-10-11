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


def location_model_train(model, av_model_fix, data_loader, optimizer, criterion_location, criterion_category, epoch, obj_rep, weight=0.5):
    model.train()
    av_model_fix.eval()
    accs = 0
    count = 0
    losses_l = 0
    losses_c = 0
    obj_rep_cuda = torch.from_numpy(obj_rep).type(torch.FloatTensor).cuda()

    for i, data in enumerate(data_loader, 0):
        if i % 200 == 0:
            print('location batch:%d' % i)
        audio_data, posi_img_data, nega_img_data = data
        audio_data, image_data, av_labels = batch_organize(audio_data, posi_img_data, nega_img_data)

        audio_data, image_data, av_labels = audio_data.type(torch.FloatTensor).cuda(), \
                                            image_data.type(torch.FloatTensor).cuda(), \
                                            av_labels.type(torch.FloatTensor).cuda()

        model.zero_grad()
        av_outputs, _, _, sounding_objects, sounding_object_local = model(image_data, audio_data, obj_rep_cuda, minimize=True)

        posi_outputs, nega_outputs = av_outputs
        posi_labels = av_labels[::2].contiguous()
        nega_labels = av_labels[1::2].contiguous()
        nega_labels = nega_labels.view(nega_labels.shape[0], 1).repeat([1, 4])
        nega_labels = nega_labels.view(-1)
        loss_location = criterion_location(posi_outputs, posi_labels) + criterion_location(nega_outputs, nega_labels)
        loss_location = 0.5 * loss_location

        with torch.no_grad():
            _,_,_,_,_, audio_class = av_model_fix(image_data, audio_data)

        audio_class = audio_class[::2]
        sounding_objects = sounding_objects[::2]

        audio_class = torch.nn.functional.softmax(audio_class.detach()/1., dim=1)
        sounding_objects = torch.nn.functional.log_softmax(sounding_objects, dim=1)

        loss_category = criterion_category(sounding_objects, audio_class)

        if epoch>1000:
            losses = loss_category
            losses.backward()
            optimizer.step()
        else:
            losses = loss_category + weight*loss_location
            losses.backward()
            optimizer.step()

        losses_l += loss_location.detach().cpu().numpy()
        losses_c += loss_category.detach().cpu().numpy()

        av_outputs = torch.cat([posi_outputs, nega_outputs], 0)
        av_labels = torch.cat([posi_labels, nega_labels], 0)

        acc = eva_metric(av_outputs.detach().cpu().numpy(), av_labels.cpu().numpy())
        accs += acc
        count += 1

    print('Location loss is %.3f, category loss is %.3f ' % ((losses_l/count), (losses_c/count)))

    return accs / count


def location_model_eva(model, data_loader, obj_rep):
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

            av_outputs, av_maps,_,_,_ = model(image_data, audio_data, obj_rep_cuda)

            posi_outputs, nega_outputs = av_outputs
            posi_labels = av_labels[::2].contiguous()
            nega_labels = av_labels[1::2].contiguous()
            nega_labels = nega_labels.view(nega_labels.shape[0], 1).repeat([1, 4])
            nega_labels = nega_labels.view(-1)

            av_outputs = torch.cat([posi_outputs, nega_outputs], 0)
            av_labels = torch.cat([posi_labels, nega_labels], 0)

            accs += eva_metric(av_outputs.detach().cpu().numpy(), av_labels.numpy())
            count += 1

    return accs / count


def main():
    parser = argparse.ArgumentParser(description='AID_PRETRAIN')
    parser.add_argument('--data_list_dir', type=str,
                        default='./data/data_indicator/music/solo')
    parser.add_argument('--data_dir', type=str, default='/home/ruiq/MUSIC/solo/')
    parser.add_argument('--mode', type=str, default='train', help='train/val/test')
    parser.add_argument('--json_file', type=str,
                        default='./data/MUSIC_label/MUSIC_solo_videos.json')
    parser.add_argument('--weight', type=float, default=0.5,
                        help='weight for location loss and category loss')
    parser.add_argument('--use_class_task', type=int, default=1, help='whether to use class task')
    parser.add_argument('--init_num', type=int, default=8, help='epoch number for initializing the location model')
    parser.add_argument('--use_pretrain', type=int, default=1, help='whether to init from ckpt')
    parser.add_argument('--ckpt_file', type=str, default='location_cluster_net_iter_006_av_class.pth', help='pretrained model name')
    parser.add_argument('--enable_img_augmentation', type=int, default=1, help='whether to augment input image')
    parser.add_argument('--enable_audio_augmentation', type=int, default=1, help='whether to augment input audio')
    parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='training batch size')
    parser.add_argument('--epoch', type=int, default=5, help='training epoch')
    parser.add_argument('--class_iter', type=int, default=3, help='training iteration for classification model')
    parser.add_argument('--gpu_ids', type=str, default='[0,1,2,3]', help='USING GPU IDS e.g.\'[0,4]\'')
    parser.add_argument('--num_threads', type=int, default=12, help='number of threads')
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--cluster', type=int, default=11)
    parser.add_argument('--mask', type=float, default=0.05)
    args = parser.parse_args()
    
    weight = args.weight

    train_dataset = MUSIC_Dataset(args)
    args_test = args
    args_test.mode = 'val'
    val_dataset = MUSIC_Dataset(args_test)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_threads)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_threads)


    # net setup
    visual_backbone = resnet18(modal='vision', pretrained=True)
    audio_backbone = resnet18(modal='audio')
    av_model = Location_Net_stage_two(visual_net=visual_backbone, audio_net=audio_backbone, cluster=args.cluster)
    if args.use_pretrain:
        PATH = args.ckpt_file
        state = torch.load(PATH)
        av_model.load_state_dict(state, strict=False)
        print(PATH)

    av_model_cuda = av_model.cuda()

    # fixed model
    visual_backbone_fix = resnet18(modal='vision', pretrained=True)
    audio_backbone_fix = resnet18(modal='audio')
    av_model_fix = Location_Net_stage_one(visual_net=visual_backbone_fix, audio_net=audio_backbone_fix, cluster=args.cluster)
    if args.use_pretrain:
        PATH = args.ckpt_file
        state = torch.load(PATH)
        av_model_fix.load_state_dict(state)
        print('loaded weights')
    av_model_fix_cuda = av_model_fix.cuda()


    obj_rep = np.load('obj_features_%.2f_%d/obj_feature_softmax_avg_fc_epoch_10_av_entire.npy' % (args.mask, args.cluster))

    loss_func_BCE_location = torch.nn.BCELoss(reduce=True)
    loss_func_BCE_category = torch.nn.KLDivLoss(reduce=True)

    optimizer = optim.Adam(params=av_model_cuda.parameters(), lr=args.learning_rate, betas=(0.9, 0.999),
                           weight_decay=0.0001)

    init_num = 0
    for e in range(args.epoch):
        print('Epoch is %03d' % e)

        train_location_acc = location_model_train(av_model_cuda, av_model_fix_cuda, train_dataloader, optimizer,
                                                  loss_func_BCE_location, loss_func_BCE_category, e, obj_rep, weight)

        eva_location_acc = location_model_eva(av_model_cuda, val_dataloader, obj_rep)

        print('train acc is %.3f eval acc is %.3f' % (train_location_acc, eva_location_acc))
        init_num += 1
        if e % 1 == 0:
            PATH = 'ckpt/stage_syn_%.2f_%d/location_cluster_net_%03d_%.3f_avg_whole.pth' % (args.mask, args.cluster, e, train_location_acc)
            torch.save(av_model_cuda.state_dict(), PATH)

if __name__ == '__main__':
    main()

