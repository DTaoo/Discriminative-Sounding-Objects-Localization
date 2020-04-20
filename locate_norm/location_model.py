import torch
import torch.nn as nn

class Location_Net_stage_one(nn.Module):
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
        self.conv_av = nn.Conv2d(1, 1, kernel_size=1, bias=False)
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
        a = torch.nn.functional.normalize(a, dim=1)
        v = torch.nn.functional.normalize(v, dim=1)
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


class Location_Net_stage_two(nn.Module):
    def __init__(self, visual_net, audio_net, use_obj_rep=True):
        super(Location_Net_stage_two, self).__init__()
        self.class_num = 11
        self.use_obj_rep = use_obj_rep

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
        self.conv_av = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        self.max_pooling_av = nn.AdaptiveMaxPool2d((1, 1))

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.v_class_layer = nn.Linear(512,11)
        self.a_class_layer = nn.Linear(512, 11)

        self.obj_pooling = nn.AdaptiveMaxPool2d((1, 1))

        self.obj_conv = nn.Conv3d(1, 1, kernel_size=1), bias=False


    def forward(self, v_input, a_input, obj_rep, minimize=False):
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
        v = torch.nn.functional.normalize(v, dim=1)
        a = torch.nn.functional.normalize(a, dim=1)
        av = torch.sum(torch.mul(v,a), 1, keepdim=True)
        av = self.conv_av(av)
        av_map = self.sigmoid(av)
        av_output = self.max_pooling_av(av_map)
        av_output = torch.flatten(av_output, 0)
        

        # classfication
        a_logits = self.a_class_layer(a_fea)

        # obj_mask
        av_map_ = av_map.repeat(1, 11, 1, 1)
        v_fea = v_fea.permute(0, 2, 3, 1)
        obj_rep = obj_rep.permute(1, 0)
        v_fea = torch.nn.functional.normalize(v_fea, dim=1)
        obj_rep = torch.nn.functional.normalize(obj_rep, dim=1)
        obj_mask = torch.matmul(v_fea, obj_rep)
        obj_mask = obj_mask.permute(0, 3, 1, 2)
        obj_mask = obj_mask.unsqueeze(1)

        obj_mask = self.obj_conv(obj_mask)
        obj_mask = torch.squeeze(obj_mask, 1)
        ############ pretrain the stage-two network without this operation firstly, then run this operation
        if minimize:
            obj_mask = torch.min(obj_mask, av_map_)
#            obj_mask = obj_mask * av_map_

        sounding_objects = self.obj_pooling(obj_mask) #sounding_obj_mask
        sounding_objects = torch.flatten(sounding_objects,1)
        sounding_objects = torch.nn.functional.softmax(sounding_objects, dim=1)
        
        obj_mask = torch.softmax(obj_mask, 1)
        return av_output, av_map, a_logits, sounding_objects, obj_mask
