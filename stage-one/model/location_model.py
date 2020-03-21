import torch
import torch.nn as nn

class Location_Net(nn.Module):
    def __init__(self, visual_net, audio_net):
        super(Location_Net, self).__init__()

        # backbone net
        self.visual_net = visual_net
        self.audio_net = audio_net

        # visual ops
        self.conv_v_1 = nn.Conv2d(512, 128, kernel_size=1)
        self.conv_v_2 = nn.Conv2d(128, 128, kernel_size=1)

        # audio ops
        self.pooling_a = nn.AdaptiveMaxPool2d((1, 1))
        self.fc_a_1 = nn.Linear(512, 128)
        self.fc_a_2 = nn.Linear(128, 128)

        # fusion ops
        self.conv_av = nn.Conv2d(1, 1, kernel_size=1)
        self.max_pooling_av = nn.AdaptiveMaxPool2d((1, 1))

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, v_input, a_input):
        # visual pathway
        v = self.visual_net(v_input)
        v = self.conv_v_1(v)
        v = self.relu(v)
        v = self.conv_v_2(v)
        v = self.relu(v)

        # audio pathway
        a = self.audio_net(a_input)
        a = self.pooling_a(a)
        a = torch.flatten(a, 1)
        a = self.fc_a_1(a)
        a = self.relu(a)
        a = self.fc_a_2(a)

        # av location
        a = torch.unsqueeze(torch.unsqueeze(a, -1), -1)
        av = torch.sum(torch.mul(v,a), 1, keepdim=True)
        av = self.conv_av(av)
        av_map = self.sigmoid(av)
        
        av_output = self.max_pooling_av(av_map)
        av_output = torch.squeeze(av_output)
        return av_output, v








class Location_Net_Video(nn.Module):
    def __init__(self, visual_net, audio_net):
        super(Location_Net_Video, self).__init__()

        # backbone net
        self.visual_net = visual_net
        self.audio_net = audio_net

        # visual ops
        self.conv_v_1 = nn.Conv2d(512, 128, kernel_size=1)
        self.conv_v_2 = nn.Conv2d(128, 128, kernel_size=1)
        self.pooling_v_t = nn.AdaptiveMaxPool3d((1, None, None))


        # audio ops
        self.pooling_a = nn.AdaptiveMaxPool2d((1, 1))
        self.fc_a_1 = nn.Linear(512, 128)
        self.fc_a_2 = nn.Linear(128, 128)

        # fusion ops
        self.conv_av = nn.Conv2d(1, 1, kernel_size=1)
        self.pooling_av = nn.AdaptiveMaxPool2d((1, 1))
        #self.pooling_av = nn.AdaptiveAvgPool2d((1, 1))

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def _img_forward(self, v_input):
        # visual pathway
        v = self.visual_net(v_input)
        v = self.conv_v_1(v)
        v = self.relu(v)
        v = self.conv_v_2(v)
        v = self.relu(v)
        return v

    def forward(self, v_input, a_input):

        v_0 = self._img_forward(v_input[:, :, 0, :, :]).unsqueeze(2)
        v_1 = self._img_forward(v_input[:, :, 1, :, :]).unsqueeze(2)
        v_2 = self._img_forward(v_input[:, :, 2, :, :]).unsqueeze(2)
        v_3 = self._img_forward(v_input[:, :, 3, :, :]).unsqueeze(2)
        
        v = torch.cat((v_0, v_1, v_2, v_3), dim=2)
        v = self.pooling_v_t(v)
        v = torch.squeeze(v)

        # audio pathway
        a = self.audio_net(a_input)
        a = self.pooling_a(a)
        a = torch.flatten(a, 1)
        a = self.fc_a_1(a)
        a = self.relu(a)
        a = self.fc_a_2(a)

        # av location
        a = torch.unsqueeze(torch.unsqueeze(a, -1), -1)
        av = torch.sum(torch.mul(v,a), 1, keepdim=True)
        av = self.conv_av(av)
        av_map = self.sigmoid(av)
        av_output = self.pooling_av(av_map)
        av_output = torch.squeeze(av_output)
        return av_output






class Location_Net_Video_Cluster(nn.Module):
    def __init__(self, visual_net, audio_net):
        super(Location_Net_Video_Cluster, self).__init__()

        self.class_layer = None

        # backbone net
        self.visual_net = visual_net
        self.audio_net = audio_net

        # visual ops
        self.conv_v_1 = nn.Conv2d(512, 128, kernel_size=1)
        self.conv_v_2 = nn.Conv2d(128, 128, kernel_size=1)
        self.pooling_v_t = nn.AdaptiveMaxPool3d((1, None, None))

        # audio ops
        self.pooling_a = nn.AdaptiveMaxPool2d((1, 1))
        self.fc_a_1 = nn.Linear(512, 128)
        self.fc_a_2 = nn.Linear(128, 128)

        # fusion ops
        self.conv_av = nn.Conv2d(1, 1, kernel_size=1)
        self.pooling_av = nn.AdaptiveMaxPool2d((1, 1))
        # self.pooling_av = nn.AdaptiveAvgPool2d((1, 1))

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.cl_sigmoid = nn.Sigmoid()
        self.cl_temporal_pooling = nn.AdaptiveMaxPool3d((1, 1, 1))

    def _img_forward(self, v):
        # visual pathway
        v = self.conv_v_1(v)
        v = self.relu(v)
        v = self.conv_v_2(v)
        v = self.relu(v)
        return v

    def forward(self, v_input, a_input):
        #v_0 = self.visual_net(v_input[:, :, 0, :, :])
        #v_1 = self.visual_net(v_input[:, :, 1, :, :])
        #v_2 = self.visual_net(v_input[:, :, 2, :, :])
        #v_3 = self.visual_net(v_input[:, :, 3, :, :])

        #v_fea = torch.cat((v_0.unsqueeze(2), v_1.unsqueeze(2), v_2.unsqueeze(2), v_3.unsqueeze(2)), dim=2)

        (B, C, T, H, W) = v_input.size()
        v = v_input.permute(0, 2, 1, 3, 4).contiguous()
        v = v.view(B*T, C, H, W)
        v = self.visual_net(v)

        (_, C, H, W) = v.size()
        v_fea = v.view(B, T, C, H, W)
        v_fea = v_fea.permute(0, 2, 1, 3, 4)
        
        if self.class_layer:
            v_fea = self.cl_temporal_pooling(v_fea)
            v_fea = torch.flatten(v_fea, 1)
            logits = self.class_layer(v_fea)
            logits = self.cl_sigmoid(logits)
            return logits
        else:
            #v_0 = self._img_forward(v_0).unsqueeze(2)
            #v_1 = self._img_forward(v_1).unsqueeze(2)
            #v_2 = self._img_forward(v_2).unsqueeze(2)
            #v_3 = self._img_forward(v_3).unsqueeze(2)

            #v = torch.cat((v_0, v_1, v_2, v_3), dim=2)
            v = self._img_forward(v)
            (_, C, H, W) = v.size()
            v = v.view(B, T, C, H, W)
            v = v.permute(0, 2, 1, 3, 4)

            v = self.pooling_v_t(v)
            v = torch.squeeze(v)

            # audio pathway
            a = self.audio_net(a_input)
            a = self.pooling_a(a)
            a = torch.flatten(a, 1)
            a = self.fc_a_1(a)
            a = self.relu(a)
            a = self.fc_a_2(a)

            # av location
            a = torch.unsqueeze(torch.unsqueeze(a, -1), -1)
            av = torch.sum(torch.mul(v, a), 1, keepdim=True)
            av = self.conv_av(av)
            av_map = self.sigmoid(av)
            av_output = self.pooling_av(av_map)
            av_output = torch.squeeze(av_output)
            return av_output, av_map, v_fea
