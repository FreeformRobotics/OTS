import torch
from torch import nn
from torch.nn import Module, Conv2d, Parameter, Softmax

# OPM适用于obj pair; OPAM适用于attention
_all__ = ["weight_init", "Obj_Attn_Block", "OAM_GRAM", "Classifier"]
    

def weight_init(module: nn.Module) -> None:
    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    if module.bias is not None:  # pyre-ignore
        nn.init.constant_(module.bias, 0)


class Obj_Attn_Block(Module):
    def __init__(self, in_dim, compress):
        super(Obj_Attn_Block, self).__init__()
        channel_in = in_dim//int(2*compress)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=channel_in, kernel_size=1)
        self.query_conv = Conv2d(in_channels=channel_in, out_channels=channel_in, kernel_size=1)
        self.key_conv = Conv2d(in_channels=channel_in, out_channels=channel_in, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1), requires_grad=True)
        self.softmax = Softmax(dim=-1)

        for layer in [self.value_conv, self.query_conv, self.key_conv]:
            weight_init(layer)

    def forward(self, x):
        m_batchsize, C, length,  _ = x.size()
        proj_value = self.value_conv(x).view(m_batchsize, -1, length)
        x = proj_value.view(m_batchsize, -1, length, 1)
        proj_query = self.query_conv(x).view(m_batchsize, -1, length).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, length)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        x = torch.bmm(proj_value, attention.permute(0, 2, 1))
        x = torch.cat((self.gamma*x, proj_value), dim=1).view(m_batchsize, -1, length, 1)
        return x


class OAM_GRAM(Module):
    def __init__(self, in_dim, one_hot_cls_num):
        super(OAM_GRAM, self).__init__()
        self.attn1 = Obj_Attn_Block(in_dim, 2)
        self.attn2 = Obj_Attn_Block(in_dim//2, 0.5)
        self.depth_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=in_dim,
                                    kernel_size=(one_hot_cls_num, 1),
                                    stride=1,
                                    padding=0,
                                    groups=in_dim)
        self.point_conv = Conv2d(in_channels=in_dim, out_channels=in_dim*2, kernel_size=1)
        self.norm = nn.BatchNorm2d(in_dim*2)
        self.relu = nn.ReLU(inplace=True)

        for layer in [self.depth_conv, self.point_conv]:
            weight_init(layer)

    def forward(self, x):
        x = self.attn1(x)
        x = self.attn2(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        x = self.norm(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        return x


class Classifier(nn.Module):
    def __init__(self, num_classes, in_dim):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(in_dim, num_classes)
        nn.init.normal_(self.fc.weight, std=0.01)

    def forward(self, x):
        out = self.fc(x)
        return out
