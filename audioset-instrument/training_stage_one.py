import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
from data.audioset import Audioset_Dataset, Audioset_AV_Classify

from model.base_model import resnet18
from model.location_model import Location_Net_stage_one

from sklearn import cluster, metrics
import numpy as np
from sklearn.preprocessing import normalize
import time

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


def location_dilation(location_map):
    location_map_dilated = np.uint8(np.zeros_like(location_map))
    for i in range(location_map.shape[1]):
        for j in range(location_map.shape[2]):
            if location_map[0, i, j] == 1:
                location_map_dilated[0, i, j] = 1
                if i > 0 and j > 0 and i < location_map.shape[1] - 1 and j < location_map.shape[2] - 1:
                    location_map_dilated[0, i + 1, j] = 1
                    location_map_dilated[0, i - 1, j] = 1
                    location_map_dilated[0, i, j - 1] = 1
                    location_map_dilated[0, i, j + 1] = 1
    return location_map_dilated


def location_model_train(model, data_loader, optimizer, criterion):
    model.train()
    accs = 0
    count = 0
    losses = 0
    batch_num = len(data_loader)
    for i, data in enumerate(data_loader, 0):
        if i % 200 == 0:
            time_str = time.asctime(time.localtime(time.time()))
            print('%s:location batch:%d/%d' % (time_str,i,batch_num))
        audio_data, posi_img_data, nega_img_data, posi_label, nega_label, _, _ = data
        audio_data, image_data, av_labels, _ = batch_organize(audio_data, posi_img_data, nega_img_data, posi_label, nega_label)

        audio_data, image_data, av_labels = audio_data.type(torch.FloatTensor).cuda(), \
                                            image_data.type(torch.FloatTensor).cuda(), \
                                            av_labels.type(torch.FloatTensor).cuda()

        optimizer.zero_grad()
        av_outputs, _, _, _ ,_ ,_ = model(image_data, audio_data)
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
            audio_data, posi_img_data, nega_img_data, posi_label, nega_label, _, _ = data
            audio_data, image_data, av_labels, _ = batch_organize(audio_data, posi_img_data, nega_img_data, posi_label, nega_label)
            audio_data, image_data = audio_data.type(torch.FloatTensor).cuda(), \
                                     image_data.type(torch.FloatTensor).cuda()

            av_outputs, _, _, _ ,_ ,_ = model(image_data, audio_data)

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
    aud_features = []
    obj_labels = []
    img_dirs = []
    aud_dirs = []
    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            audio_data, posi_img_data, nega_img_data, posi_label, nega_label, posi_img_dir, posi_aud_dir = data

            audio_data, image_data = audio_data.type(torch.FloatTensor).cuda(), \
                                     posi_img_data.type(torch.FloatTensor).cuda()

            _, location_map, v_fea, a_fea, _, _ = model(image_data, audio_data)

            location_map = location_map.detach().cpu().numpy()
            v_fea = v_fea.detach().cpu().numpy()
            a_fea = a_fea.detach().cpu().numpy()

            for j in range(location_map.shape[0]):
                obj_mask = np.uint8(location_map[j] > 0.05)
                obj_mask = location_dilation(obj_mask)

                img_rep = v_fea[j, :, :, :]  # such as 512 x 14 x 14
                obj_rep = img_rep * obj_mask
            ######################################################
                obj_features.append(np.mean(obj_rep, (1,2)))
                img_features.append(np.mean(img_rep, (1,2)))
                aud_features.append(a_fea[j,:])
                obj_labels.append(posi_label[j].numpy())   ########

            img_dirs.extend(list(posi_img_dir))
            aud_dirs.extend(list(posi_aud_dir))

    return np.asarray(obj_features), np.asarray(img_features), np.asarray(aud_features),  np.asarray(obj_labels), img_dirs, aud_dirs


def feature_clustering(data, label, val_data):
    data = normalize(data, norm='l2')
    val_data = normalize(val_data, norm='l2')
    labels = []
    for i in range(label.shape[0]):
        if label[i] not in labels:
            labels.append(label[i])
    count = len(labels)
    #print(count)
    kmeans = cluster.KMeans(n_clusters=count, random_state=0).fit(data)
    cluster_label = kmeans.labels_
    score = metrics.normalized_mutual_info_score(label, cluster_label)
    val_label = kmeans.predict(val_data)

    return cluster_label, score, val_label


def class_model_train(model, data_loader, optimizer, criterion, args, device):
    model.v_class_layer = torch.nn.Linear(512, 13)
    model.v_class_layer.weight.data.normal_(0, 0.01)
    model.v_class_layer.bias.data.zero_()
    #model.v_class_layer.cuda()
    model.v_class_layer = torch.nn.DataParallel(model.v_class_layer)
    model.v_class_layer.to(device)


    model.a_class_layer = torch.nn.Linear(512, 13)
    model.a_class_layer.weight.data.normal_(0, 0.01)
    model.a_class_layer.bias.data.zero_()
    #model.a_class_layer.cuda()
    model.a_class_layer = torch.nn.DataParallel(model.a_class_layer)
    model.a_class_layer.to(device)

    model.train()

    params = list(model.v_class_layer.parameters()) + list(model.a_class_layer.parameters())
    optimizer_cl = torch.optim.Adam(params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0001)

    for j in range(args.class_iter):
        count = 0
        losses = 0
        for i, data in enumerate(data_loader, 0):
            aud, img, label = data
            img, aud, label = img.type(torch.FloatTensor).cuda(), \
                              aud.type(torch.FloatTensor).cuda(), \
                              label.type(torch.LongTensor).cuda()

            _, _, _, _, v_logits, a_logits = model(img, aud)
            loss_v = criterion(v_logits, label)
            loss_a = criterion(a_logits, label)
            loss = loss_a + loss_v

            optimizer.zero_grad()
            optimizer_cl.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer_cl.step()

            losses += loss.detach().cpu().numpy()
            count += 1

        print('class loss is %.3f ' % (losses / count))


def class_model_val(model, data_loader):
    model.eval()
    a_count = 0
    v_count = 0
    count = 0
    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            aud, img, label = data
            img, aud = img.type(torch.FloatTensor).cuda(), \
                       aud.type(torch.FloatTensor).cuda()

            _, _, _, _, v_logits, a_logits = model(img, aud)

            label = label.numpy()
            v_logits = np.argmax(v_logits.cpu().numpy())
            a_logits = np.argmax(a_logits.cpu().numpy())

            v_count += np.sum(v_logits == label)#label_[:, -1])
            a_count += np.sum(a_logits == label)#label_[:, -1])
            count += label.shape[0]
    print('class: audio acc: %.3f, image acc: %.3f' % ((a_count/count), (v_count/count)))



def main():
    parser = argparse.ArgumentParser(description='AID_PRETRAIN')
    parser.add_argument('--data_list_dir', type=str,
                        default='/mnt/home/hudi/location_sound/audioset-exp/data/audioset_single')
    parser.add_argument('--data_dir', type=str, default='/mnt/scratch/hudi/audioset-instrument/')
    parser.add_argument('--mode', type=str, default='train', help='train/val/test')
    parser.add_argument('--use_class_task', type=int, default=1, help='whether to use class task')
    parser.add_argument('--init_num', type=int, default=10, help='epoch number for initializing the location model')
    parser.add_argument('--use_pretrain', type=int, default=0, help='whether to init from ckpt')
    parser.add_argument('--class_iter', type=int, default=3, help='training iteration for classification model')
    parser.add_argument('--ckpt_file', type=str, default='location_cluster_net_010_0.686.pth', help='pretrained model name')
    parser.add_argument('--enable_img_augmentation', type=int, default=1, help='whether to augment input image')
    parser.add_argument('--enable_audio_augmentation', type=int, default=1, help='whether to augment input audio')
    parser.add_argument('--batch_size', type=int, default=32*8, help='training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='training batch size')
    parser.add_argument('--epoch', type=int, default=2000, help='training epoch')
    parser.add_argument('--gpu_ids', type=str, default='[0,1,2,3,4,5,6,7]', help='USING GPU IDS e.g.\'[0,4]\'')
    parser.add_argument('--num_threads', type=int, default=100, help='number of threads')
    parser.add_argument('--seed', type=int, default=10)
    args = parser.parse_args()

    if args.init_num != 0 and args.use_pretrain:
        import sys
        print('If use ckpt, do not recommend to set init_num to 0.')
        sys.exit()

    train_dataset = Audioset_Dataset(args.data_dir, 'single_train_data.pkl', args)
    val_dataset = Audioset_Dataset(args.data_dir, 'single_val_data.pkl', args)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_threads)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_threads)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # net setup
    visual_backbone = resnet18(modal='vision')
    audio_backbone = resnet18(modal='audio')
    av_model = Location_Net_stage_one(visual_net=visual_backbone, audio_net=audio_backbone)

    print("Let's use", torch.cuda.device_count(), "GPUs!")
    av_model_cuda = torch.nn.DataParallel(av_model)
    av_model_cuda.to(device)

    #av_model_cuda = av_model.cuda()

    if args.use_pretrain:
        PATH = os.path.join('ckpt/stage_one/', args.ckpt_file)
        state = torch.load(PATH)
        av_model_cuda.load_state_dict(state)
        print('loaded weights')

    loss_func_BCE_location = torch.nn.BCELoss(reduce=True)
    loss_func_BCE_class = torch.nn.CrossEntropyLoss()  # torch.nn.BCELoss(reduce=True)

    params = list(av_model_cuda.parameters())
    optimizer_location = optim.Adam(params=params[:-4], lr=args.learning_rate, betas=(0.9, 0.999),
                           weight_decay=0.0001)

    init_num = 0
    for e in range(args.epoch):

        ############################### location training #################################
        print('Epoch is %03d' % e)
        train_location_acc = location_model_train(av_model_cuda, train_dataloader, optimizer_location, loss_func_BCE_location)
        eva_location_acc = location_model_eva(av_model_cuda, val_dataloader)

        ############################### classification training #################################
        if init_num > args.init_num and args.use_class_task:

            obj_features, img_features, aud_features, labels, img_dirs, aud_dirs = extract_feature(av_model_cuda, train_dataloader)
            val_obj_features, val_img_features, val_aud_features, _, val_img_dirs, val_aud_dirs = extract_feature(av_model_cuda, val_dataloader)

            pseudo_label, nmi_score, val_pseudo_label = feature_clustering(obj_features, labels, val_obj_features)
            print('obj_NMI is %.3f' % nmi_score)

            obj_fea = []
            for i in range(13):
                cur_idx = pseudo_label == i
                cur_fea = obj_features[cur_idx]
                obj_fea.append(np.mean(cur_fea, 0))
            np.save('obj_feature_softmax_avg_fc.npy', obj_fea)

            class_dataset = Audioset_AV_Classify(img_dirs, aud_dirs, pseudo_label, args)
            class_dataloader = DataLoader(dataset=class_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_threads)
            class_dataset_val = Audioset_AV_Classify(val_img_dirs, val_aud_dirs, val_pseudo_label, args)
            class_dataloader_val = DataLoader(dataset=class_dataset_val, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.num_threads)

            class_model_train(av_model_cuda, class_dataloader, optimizer_location, loss_func_BCE_class, args, device)
            class_model_val(av_model_cuda, class_dataloader_val)

        print('train acc is %.3f, val acc is %.3f' % (train_location_acc, eva_location_acc))
        init_num += 1
        if e % 1 == 0:
            PATH = 'ckpt/stage_one/location_cluster_net_%03d_%.3f.pth' % (e, eva_location_acc)
            torch.save(av_model_cuda.state_dict(), PATH)

if __name__ == '__main__':
    main()
