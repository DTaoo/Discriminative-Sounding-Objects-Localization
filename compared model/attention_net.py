import torch
import torch.nn as nn

class Attention_Net(nn.Module):
    def __init__(self, visual_net, audio_net):
        super(Attention_Net, self).__init__()

        # backbone net
        self.visual_net = visual_net
        self.audio_net = audio_net
        self.pooling = nn.AdaptiveMaxPool2d((1, 1))

        # visual ops
        self.fc_v_1 = nn.Linear(512, 512)
        self.fc_v_2 = nn.Linear(512, 512)

        # audio ops
        self.pooling_a = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_a_1 = nn.Linear(512, 512)
        self.fc_a_2 = nn.Linear(512, 512)

        self.relu = nn.ReLU(inplace=True)
        # fusion ops
        self.fc_av = nn.Linear(1, 2)


    def forward(self, v_input, a_input):
        # visual pathway
        v_fea = self.visual_net(v_input)
        (B, C, H, W) = v_fea.size()
        v_fea = v_fea.view(B, C, H * W)
        v_fea = v_fea.permute(0, 2, 1)  # B, HxW, C
        v = nn.functional.normalize(v_fea, dim=2)  # B, HxW, C

        # audio pathway
        a_fea = self.audio_net(a_input)
        a_fea = self.pooling(a_fea)
        a_fea = torch.flatten(a_fea, 1)
        a = self.fc_a_1(a_fea)
        a = self.relu(a)
        a = self.fc_a_2(a)
        a = self.relu(a)
        a = a.unsqueeze(1)
        a = nn.functional.normalize(a, dim=2)

        # attention
        a_ = a.permute(0, 2, 1) # B, C, 1
        av_atten = torch.bmm(v, a_)
        av_atten = nn.functional.softmax(av_atten, dim=1)  # B, HxW, 1
        #print(av_atten.shape)
        z = torch.sum(av_atten*v, dim=1) # BxC
        #print(z.shape)

        v = self.fc_v_1(z)
        v = self.relu(v)
        v = self.fc_v_2(v)
        v = self.relu(v)

        av = torch.norm(v - a_fea, dim=1, keepdim=True)

        return av, av_atten, v
