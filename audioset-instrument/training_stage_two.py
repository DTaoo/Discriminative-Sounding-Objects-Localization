import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
from data.audioset import Audioset_Dataset, Audioset_AV_Classify

from model.base_model import resnet18
from model.location_model import Location_Net_stage_two

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


def location_model_train(model, av_model_fix, data_loader, optimizer, criterion_location, criterion_class, epoch, obj_rep):
    model.train()
    av_model_fix.eval()
    accs = 0
    count = 0
    losses_g = 0
    losses_l = 0
    obj_rep_cuda = torch.from_numpy(obj_rep).type(torch.FloatTensor).cuda()
    for i, data in enumerate(data_loader, 0):
        if i % 200 == 0:
            print('location batch:%d' % i)
        audio_data, posi_img_data, nega_img_data, _,_,_,_ = data
        audio_data, image_data, av_labels = batch_organize(audio_data, posi_img_data, nega_img_data)

        audio_data, image_data, av_labels = audio_data.type(torch.FloatTensor).cuda(), \
                                            image_data.type(torch.FloatTensor).cuda(), \
                                            av_labels.type(torch.FloatTensor).cuda()

        model.zero_grad()
        av_outputs, _, _, sounding_objects, sounding_object_local = model(image_data, audio_data, obj_rep_cuda)
        loss_global = criterion_location(av_outputs, av_labels)

        with torch.no_grad():
            _, _, audio_class, _, _ = av_model_fix(image_data, audio_data, obj_rep_cuda)

        audio_class = audio_class[::2]
        sounding_objects = sounding_objects[::2]

        pseudo_label = torch.zeros_like(audio_class)
        idx = torch.argsort(audio_class, axis=1)
        pseudo_label[np.arange(audio_class.shape[0]), idx[:, -2]] = 1
        pseudo_label[np.arange(audio_class.shape[0]), idx[:, -1]] = 1
        pseudo_label = pseudo_label.cuda()

        #sounding_objects = torch.nn.functional.log_softmax(sounding_objects/5, dim=1)
        audio_class = torch.nn.functional.softmax(audio_class.detach()/0.7, dim=1)

        loss_local = criterion_class(sounding_objects, audio_class)

        if epoch<50:
            losses = loss_local
            losses.backward()
            optimizer.step()
        else:
            losses = loss_local + loss_global
            optimizer.step()
            optimizer.step()

        losses_g += loss_global.detach().cpu().numpy()
        losses_l += loss_local.detach().cpu().numpy()

        acc = eva_metric(av_outputs.detach().cpu().numpy(), av_labels.cpu().numpy())
        accs += acc
        count += 1

    print('Global matching loss is %.3f, local matching loss is %.3f ' % ((losses_g/count), (losses_l/count)))

    return accs / count



def main():
    parser = argparse.ArgumentParser(description='AID_PRETRAIN')
    parser.add_argument('--data_list_dir', type=str,
                        default='/mnt/home/hudi/location_sound/audioset-exp/data/audioset_single')
    parser.add_argument('--data_dir', type=str, default='/mnt/scratch/hudi/audioset-instrument/')
    parser.add_argument('--mode', type=str, default='train', help='train/val/test')
    parser.add_argument('--use_class_task', type=int, default=1, help='whether to use class task')
    parser.add_argument('--init_num', type=int, default=0, help='epoch number for initializing the location model')
    parser.add_argument('--use_pretrain', type=int, default=0, help='whether to init from ckpt')
    parser.add_argument('--class_iter', type=int, default=3, help='training iteration for classification model')
    parser.add_argument('--ckpt_file', type=str, default='location_net_009_0.665.pth', help='pretrained model name')
    parser.add_argument('--enable_img_augmentation', type=int, default=1, help='whether to augment input image')
    parser.add_argument('--enable_audio_augmentation', type=int, default=1, help='whether to augment input audio')
    parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='training batch size')
    parser.add_argument('--epoch', type=int, default=2000, help='training epoch')
    parser.add_argument('--gpu_ids', type=str, default='[0,1,2,3]', help='USING GPU IDS e.g.\'[0,4]\'')
    parser.add_argument('--num_threads', type=int, default=12, help='number of threads')
    parser.add_argument('--seed', type=int, default=10)
    args = parser.parse_args()

    if args.init_num != 0 and args.use_pretrain:
        import sys
        print('If use ckpt, do not recommend to set init_num to 0.')
        sys.exit()

    train_dataset = Audioset_Dataset(args.data_dir, 'multi_train.pkl', args)
    val_dataset = Audioset_Dataset(args.data_dir, 'multi_val.pkl', args)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_threads)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_threads)

    # net setup
    visual_backbone = resnet18(modal='vision')
    audio_backbone = resnet18(modal='audio')
    av_model = Location_Net_stage_two(visual_net=visual_backbone, audio_net=audio_backbone)

    if args.use_pretrain:
        PATH = os.path.join('ckpt/stage_one/', args.ckpt_file)
        state = torch.load(PATH)
        av_model.load_state_dict(state, strict=False)
    av_model_cuda = av_model.cuda()

    # fixed model
    visual_backbone_fix = resnet18(modal='vision')
    audio_backbone_fix = resnet18(modal='audio')
    av_model_fix = Location_Net_stage_two(visual_net=visual_backbone_fix, audio_net=audio_backbone_fix)

    if args.use_pretrain:
        PATH = os.path.join('ckpt/stage_one/', args.ckpt_file)
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
    for e in range(args.epoch):
        print('Epoch is %03d' % e)

        train_location_acc = location_model_train(av_model_cuda, av_model_fix_cuda, train_dataloader, optimizer,
                                                  loss_func_BCE_location, loss_func_BCE_class, e, obj_rep)

        if e % 5 == 0:
            PATH = 'ckpt/stage_two/audioset_pretrain_%03d_%.3f.pth' % (e, train_location_acc)
            torch.save(av_model_cuda.state_dict(), PATH)

if __name__ == '__main__':
    main()