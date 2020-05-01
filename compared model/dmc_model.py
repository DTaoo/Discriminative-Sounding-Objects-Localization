import torch
import torch.nn as nn
import random

class Cluster_layer(nn.Module):
    def __init__(self, input_dim = 512, num_cluster=2, iters=10, beta=-10, **kwargs):
        super(Cluster_layer, self).__init__()
        self.input_dim = input_dim
        self.num_cluster = num_cluster
        self.iters = iters
        self.beta = beta
        self.epsilon = torch.tensor(1e-10).type(torch.FloatTensor).cuda()

    def forward(self, u_vecs):

        (batch_size, input_num, feature_dim) = u_vecs.size()

        ini_interval = int(196/self.num_cluster)  #

        o = torch.unsqueeze(u_vecs[:, 50, :], dim=1)
        count = 1
        while(self.num_cluster-count > 0):
            current_o = torch.unsqueeze(u_vecs[:, ini_interval*count, :], dim=1)  #ini_interval*count
            o = torch.cat([o, current_o], dim=1)
            count += 1

        for i in range(self.iters):
            nx = torch.sum(o**2, dim=2, keepdim=True)
            ny = torch.sum(u_vecs**2, dim=2, keepdim=True)

            qq = nx - 2 * torch.bmm(o, u_vecs.permute(0,2,1)) + ny.permute(0,2,1)
            b = torch.sqrt(torch.max(qq, self.epsilon))
            c = nn.functional.softmax(self.beta*b, dim=1)   # assignments  [None, output_num_capsule, input_num_capsule]
            o = torch.bmm(c, u_vecs)     # cluster centers  [None, num_cluster, dim_cluster]
            weights = torch.sum(c, dim=2, keepdim=True)
            o = o / weights

        return o, c


class DMC_NET(nn.Module):
    def __init__(self, visual_net, audio_net, v_cluster_num = 2, a_cluster_num = 1):
        super(DMC_NET, self).__init__()

        # backbone net
        self.visual_net = visual_net
        self.audio_net = audio_net
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

        # visual ops
        self.fc_v_1 = nn.Linear(512, 512)
        self.fc_v_2 = nn.Linear(128, 128)

        # audio ops
        self.pooling_a = nn.AdaptiveMaxPool2d((1, 1))
        self.fc_a_1 = nn.Linear(512, 512)
        self.fc_a_2 = nn.Linear(128, 128)

        self.relu = nn.ReLU(inplace=True)
        # fusion ops
        self.fc_av = nn.Linear(1, 2)

        self.v_clustering = Cluster_layer(num_cluster=v_cluster_num)
        self.a_clustering = Cluster_layer(num_cluster=a_cluster_num)
        self.epsilon = torch.tensor(1e-10).type(torch.FloatTensor).cuda()


    def forward(self, v_input, a_input):
        # visual pathway
        v_fea = self.visual_net(v_input)
        (B, C, H, W) = v_fea.size()
        v_fea = v_fea.view(B, C, H*W)
        v_fea = v_fea.permute(0,2,1)
        v_fea = self.fc_v_1(v_fea)
        v_centers, v_assign = self.v_clustering(v_fea)

        # audio pathway
        a_fea = self.audio_net(a_input)
        (B, C, H, W) = a_fea.size()
        a_fea = a_fea.view(B, C, H*W)
        a_fea = a_fea.permute(0,2,1)
        a_fea = self.fc_a_1(a_fea)
        a_centers, a_assign = self.a_clustering(a_fea)

        v_centers_ = torch.sum(v_centers ** 2, dim=2, keepdim=True)
        a_centers_ = torch.sum(a_centers ** 2, dim=2, keepdim=True)

        distance_ = torch.sqrt(torch.max(v_centers_ - 2 * torch.bmm(v_centers, a_centers.permute(0, 2, 1)) + a_centers_.permute(0, 2, 1), self.epsilon))
        distance = torch.min(distance_, dim=1)
        distance = distance.values

        return distance, v_assign, distance_
