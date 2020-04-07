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

        self.v_class_layer = nn.Linear(512, 11)
        self.a_class_layer = nn.Linear(512, 11)
        self.pooling_v = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, v_input, a_input):
        # visual pathway
        v_fea = self.visual_net(v_input)
        v = self.conv_v_1(v_fea)
        v = self.relu(v)
        v = self.conv_v_2(v)

        # audio pathway
        a = self.audio_net(a_input)
        a = self.pooling_a(a)
        a_fea = torch.flatten(a, 1)
        a = self.fc_a_1(a_fea)
        a = self.relu(a)
        a = self.fc_a_2(a)

        # av location
        a = torch.unsqueeze(torch.unsqueeze(a, -1), -1)
        av = torch.sum(torch.mul(v,a), 1, keepdim=True)
        av = self.conv_av(av)
        av_map = self.sigmoid(av)
        av_output = self.max_pooling_av(av_map)
        av_output = torch.squeeze(av_output)

        # classfication
        v_cla = self.pooling_v(v_fea)
        v_cla = torch.flatten(v_cla, 1)
        v_logits = self.v_class_layer(v_cla)
        a_logits = self.a_class_layer(a_fea)

        return av_output, av_map, v_fea, a_fea, v_logits, a_logits