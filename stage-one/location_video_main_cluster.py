import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
from data.MUSIC_dataset import MUSIC_Dataset_video, MUSIC_Video_Classify

from model.base_model import resnet18
from model.location_model import Location_Net_Video_Cluster

from sklearn import cluster, metrics
import numpy as np
from sklearn.preprocessing import normalize


def batch_organize(audio_data, posi_img_data, nega_img_data, posi_label, nega_label):
    batch_audio_data = torch.zeros(audio_data.shape[0] * 2, audio_data.shape[1], audio_data.shape[2],
                                   audio_data.shape[3])
    batch_image_data = torch.zeros(posi_img_data.shape[0] * 2, posi_img_data.shape[1], posi_img_data.shape[2],
                                   posi_img_data.shape[3], posi_img_data.shape[4])
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
    for i, data in enumerate(data_loader, 0):
        if i % 200 == 0:
            print('location batch:%d' % i)
        audio_data, posi_img_data, nega_img_data, posi_label, nega_label, _ = data
        audio_data, image_data, av_labels, _ = batch_organize(audio_data, posi_img_data, nega_img_data, posi_label,
                                                              nega_label)

        audio_data, image_data, av_labels = audio_data.type(torch.FloatTensor).cuda(), \
                                            image_data.type(torch.FloatTensor).cuda(), \
                                            av_labels.type(torch.FloatTensor).cuda()

        optimizer.zero_grad()
        av_outputs, _, _ = model(image_data, audio_data)
        loss = criterion(av_outputs, av_labels)
        loss.backward()
        optimizer.step()

        losses += loss.detach().cpu().numpy()
        acc = eva_metric(av_outputs.detach().cpu().numpy(), av_labels.cpu().numpy())
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
            audio_data, posi_img_data, nega_img_data, posi_label, nega_label, _ = data
            audio_data, image_data, av_labels, _ = batch_organize(audio_data, posi_img_data, nega_img_data, posi_label,
                                                                  nega_label)
            audio_data, image_data = audio_data.type(torch.FloatTensor).cuda(), \
                                     image_data.type(torch.FloatTensor).cuda()

            av_outputs, _, _ = model(image_data, audio_data)

            accs += eva_metric(av_outputs.detach().cpu().numpy(), av_labels.numpy())
            count += 1

    return accs / count


def extract_feature(model, data_loader):
    print('extracting features...')
    model.eval()
    accs = 0
    num = len(data_loader.dataset)
    count = 0
    obj_features = []
    img_features = []
    obj_labels = []
    img_dirs = []
    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            audio_data, posi_img_data, nega_img_data, posi_label, nega_label, posi_dir = data
            audio_data, image_data, av_labels, class_label = batch_organize(audio_data, posi_img_data, nega_img_data,
                                                                            posi_label, nega_label)
            audio_data, image_data = audio_data.type(torch.FloatTensor).cuda(), \
                                     image_data.type(torch.FloatTensor).cuda()

            _, location_map, v_fea = model(image_data, audio_data)

            location_map = location_map.detach().cpu().numpy()
            v_fea = v_fea.detach().cpu().numpy()

            for j in range(location_map.shape[0]):
                if av_labels[j].cpu().numpy() == 0:
                    continue
                obj_mask = np.uint8(location_map[j] > 0.05)
                obj_mask = location_dilation(obj_mask)

                current_obj_rep = []
                current_img_rep = []
                for idx in range(4):
                    img_rep = v_fea[j, :, idx, :, :]  # such as batch x 512 x 4 x 14 x 14
                    obj_rep = img_rep * obj_mask
                    obj_rep = np.mean(obj_rep, (1, 2))
                    current_obj_rep.append(obj_rep)
                    current_img_rep.append(np.mean(img_rep, (1, 2)))

                obj_features.append(np.mean(current_obj_rep, 0))
                img_features.append(np.mean(current_img_rep, 0))
                obj_labels.append(class_label[j].numpy())
            img_dirs.extend(list(posi_dir))

    return np.asarray(obj_features), np.asarray(img_features), np.asarray(obj_labels), img_dirs


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

    return cluster_label, score


def class_model_train(model, data_loader, optimizer, criterion, args):
    model.class_layer = torch.nn.Linear(512, 11)
    model.class_layer.weight.data.normal_(0, 0.01)
    model.class_layer.bias.data.zero_()
    model.class_layer.cuda()

    model.train()

    optimizer_tl = torch.optim.Adam(model.class_layer.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0001)

    for j in range(args.class_iter):
        count = 0
        losses = 0
        for i, data in enumerate(data_loader, 0):
            # if i%200==0:
            #    print('class batch:%d' % i)
            img, label = data
            aud = torch.zeros_like(img)
            img, aud, label = img.type(torch.FloatTensor).cuda(), \
                              aud.type(torch.FloatTensor).cuda(), \
                              label.type(torch.LongTensor).cuda()

            predict_logits = model(img, aud)
            loss = criterion(predict_logits, label)
            optimizer.zero_grad()
            optimizer_tl.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer_tl.step()

            losses += loss.detach().cpu().numpy()
            count += 1

        print('class loss is %.3f ' % (losses / count))


def main():
    parser = argparse.ArgumentParser(description='AID_PRETRAIN')
    parser.add_argument('--data_list_dir', type=str,
                        default='/mnt/home/hudi/location_sound/data/data_indicator/music/solo')
    parser.add_argument('--data_dir', type=str, default='/mnt/scratch/hudi/MUSIC/solo/')
    parser.add_argument('--mode', type=str, default='train', help='train/val/test')
    parser.add_argument('--json_file', type=str,default='/mnt/home/hudi/location_sound/data/MUSIC_label/MUSIC_solo_videos.json')
    
    parser.add_argument('--use_class_task', type=int, default=1, help='whether to use class task')
    parser.add_argument('--init_num', type=int, default=8, help='epoch number for initializing the location model')
    parser.add_argument('--use_pretrain', type=int, default=0, help='whether to init from ckpt')
    parser.add_argument('--ckpt_file', type=str, default='location_net_009_0.665.pth', help='pretrained model name')
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

    train_list_file = os.path.join(args.data_list_dir, 'solo_pairs_train.txt')
    val_list_file = os.path.join(args.data_list_dir, 'solo_pairs_val.txt')
    test_list_file = os.path.join(args.data_list_dir, 'solo_pairs_test.txt')

    train_dataset = MUSIC_Dataset_video(args.data_dir, train_list_file, args)
    val_dataset = MUSIC_Dataset_video(args.data_dir, val_list_file, args)
    test_dataset = MUSIC_Dataset_video(args.data_dir, test_list_file, args)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_threads)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_threads)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_threads)

    # net setup
    visual_backbone = resnet18(modal='vision')
    audio_backbone = resnet18(modal='audio')
    av_model = Location_Net_Video_Cluster(visual_net=visual_backbone, audio_net=audio_backbone)

    if args.use_pretrain:
        PATH = os.path.join('ckpt/video/', args.ckpt_file)
        state = torch.load(PATH)
        av_model.load_state_dict(state)
    av_model_cuda = av_model.cuda()

    loss_func_BCE_location = torch.nn.BCELoss(reduce=True)
    loss_func_BCE_class = torch.nn.CrossEntropyLoss()  # torch.nn.BCELoss(reduce=True)

    optimizer = optim.Adam(params=av_model_cuda.parameters(), lr=args.learning_rate, betas=(0.9, 0.999),
                                    weight_decay=0.0001)

    init_num = 0
    for e in range(args.epoch):
        print('Epoch is %03d' % e)
        # av_model_cuda.class_layer = None
        # train_location_acc = location_model_train(av_model_cuda, train_dataloader, optimizer_location, loss_func_BCE_location)
        # eva_location_acc = location_model_eva(av_model_cuda, val_dataloader)

        obj_features, img_features, labels, img_dirs = extract_feature(av_model_cuda, train_dataloader)
        pseudo_label, obj_nmi_score = feature_clustering(obj_features, labels)
        _, img_nmi_score = feature_clustering(img_features, labels)

        if init_num > args.init_num and args.use_class_task:
            # pseudo_label_ = np.zeros((pseudo_label.size, 11))           # here for multi-label classification
            # pseudo_label_[np.arange(pseudo_label.size), pseudo_label] = 1

            class_dataset = MUSIC_Video_Classify(img_dirs, pseudo_label, args)
            class_dataloader = DataLoader(dataset=class_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_threads)

            class_model_train(av_model_cuda, class_dataloader, optimizer, loss_func_BCE_class, args)

        av_model_cuda.class_layer = None
        train_location_acc = location_model_train(av_model_cuda, train_dataloader, optimizer,
                                                  loss_func_BCE_location)
        eva_location_acc = location_model_eva(av_model_cuda, val_dataloader)

        print('train acc is %.3f, val acc is %.3f, obj_NMI is %.3f, img_NMI is %.3f' % (train_location_acc, eva_location_acc, obj_nmi_score, img_nmi_score))
        init_num += 1
        if e % 3 == 0:
            PATH = 'ckpt/video_cluster/location_cluster_net_softmax_%03d_%.3f_%.3f.pth' % (e, eva_location_acc, obj_nmi_score)
            #torch.save(av_model_cuda.state_dict(), PATH)


if __name__ == '__main__':
    main()


