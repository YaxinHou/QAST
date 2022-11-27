import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Sequential, Sigmoid, Parameter


def Truncated_normal(a, b, mean=0, std=1):
    size = (a, b)
    tensor = torch.zeros(size)
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2.5) & (tmp > -2.5)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


def Label_sampel(batch_size, n_class):
    label = torch.LongTensor(batch_size, 1).random_() % n_class
    one_hot = torch.zeros(batch_size, n_class).scatter_(1, label, 1)
    return label.squeeze(1).reshape(batch_size, ).float(), one_hot


class ConditionalNorm(nn.Module):
    def __init__(self, in_channel, n_class):
        super(ConditionalNorm, self).__init__()

        self.bn = BatchNorm1d(in_channel)
        self.fc = Linear(n_class, 20)

        self.embed = Linear(60, in_channel * 2)
        self.embed.weight.data[:, :in_channel] = 1
        self.embed.weight.data[:, in_channel:] = 0

    def forward(self, input, em_lable):
        a, b = em_lable.shape
        con1 = self.fc(em_lable)
        con2 = Truncated_normal(a, 40)
        con = torch.cat([con1, con2], 1)
        out = self.bn(input)
        embed = self.embed(con)
        gamma, beta = embed.chunk(2, 1)
        out = gamma * out + beta
        return out


class Residual(nn.Module):
    def __init__(self, i, o, n_class):
        super(Residual, self).__init__()
        self.fc1 = Linear(i, o)
        self.fc2 = Linear(o, int(o / 2))
        self.fc3 = Linear(int(o / 2), o)
        self.bn = ConditionalNorm(o, n_class)
        self.lerelu = LeakyReLU(0.2)

    def forward(self, input, em_lable):
        out1 = self.fc1(input)
        out2 = self.fc2(out1)
        out3 = self.fc3(out2)
        out = self.bn(out1 + out3, em_lable)
        out = self.lerelu(out)
        return out


class Block(nn.Module):
    def __init__(self, i, o):
        super(Block, self).__init__()
        self.fc1 = Linear(i, o)
        self.fc2 = Linear(o, int(o / 2))
        self.fc3 = Linear(int(o / 2), o)
        self.lerelu = LeakyReLU(0.2)
        self.dr = Dropout(0.5)

    def forward(self, input):
        out1 = self.fc1(input)
        out2 = self.fc2(out1)
        out3 = self.fc3(out2)
        out = self.lerelu(out1 + out3)
        out = self.dr(out)
        return out


class Attention(nn.Module):
    def __init__(self, i, o):
        super(Attention, self).__init__()
        self.fc1 = Linear(i, o)
        self.fc2 = Linear(i, o // 4)
        self.fc3 = Linear(i, o // 4)
        self.gamma = Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, input):
        q = self.fc1(input)
        k = self.fc2(input)
        v = self.fc3(input)

        B, W = q.size()
        q = q.view(B, 1, 1 * W)  # query
        k = k.view(B, 1, 1 * W // 4)  # key
        v = v.view(B, 1, 1 * W // 4)  # value

        # attention weights
        w = F.softmax(torch.bmm(q.transpose(1, 2), k), -1)
        # attend and project
        o = torch.bmm(v, w.transpose(1, 2)).view(B, W)
        return self.gamma * o + input


class Discriminator(nn.Module):

    def __init__(self, input_dim):
        super(Discriminator, self).__init__()

        dim = input_dim
        seq = []
        if input_dim <= 64:
            seq += [Attention(dim, dim)]
            seq += [Block(dim, 32)]
            seq += [Block(32, 16)]
            seq += [Linear(16, 8)]
        elif input_dim <= 128:
            seq += [Attention(dim, dim)]
            seq += [Block(dim, 64)]
            seq += [Block(64, 32)]
            seq += [Block(32, 16)]
            seq += [Linear(16, 8)]
        elif input_dim <= 256:
            seq += [Attention(dim, dim)]
            seq += [Block(dim, 128)]
            seq += [Block(128, 64)]
            seq += [Block(64, 32)]
            seq += [Block(32, 16)]
            seq += [Linear(16, 8)]
        else:
            seq += [Attention(dim, dim)]
            seq += [Block(dim, 256)]
            seq += [Block(256, 128)]
            seq += [Block(128, 64)]
            seq += [Block(64, 32)]
            seq += [Block(32, 16)]
            seq += [Linear(16, 8)]

        dim = 8
        seq += [Linear(dim, 1), Sigmoid()]
        self.seq = Sequential(*seq)

    def forward(self, input, label=None):
        if label is not None:
            input = torch.cat((input, label), -1)
        output = self.seq(input)
        return output


class Generator(nn.Module):
    def __init__(self, indim, data_dim, n_class):
        super(Generator, self).__init__()
        dim = indim
        if indim <= 64:
            self.block1 = Residual(dim, 64, n_class)
            self.block2 = Residual(64, 128, n_class)
            self.block3 = Residual(128, 256, n_class)
            self.block4 = Residual(256, 512, n_class)
            self.block5 = Residual(512, 256, n_class)
            self.block6 = Residual(256, 128, n_class)
            self.block7 = Residual(128, 64, n_class)
            self.block7_8 = Attention(64, 64)
            self.block8 = Linear(64, data_dim)
        elif indim <= 128:
            self.block1 = Residual(dim, 128, n_class)
            self.block2 = Residual(128, 256, n_class)
            self.block3 = Residual(256, 512, n_class)
            self.block4 = Residual(512, 256, n_class)
            self.block5 = Residual(256, 128, n_class)
            self.block5_6 = Attention(128, 128)
            self.block6 = Linear(128, data_dim)
        elif indim <= 256:
            self.block1 = Residual(dim, 256, n_class)
            self.block2 = Residual(256, 512, n_class)
            self.block3 = Residual(512, 256, n_class)
            self.block3_4 = Attention(256, 256)
            self.block4 = Linear(256, data_dim)
        else:
            self.block1 = Residual(dim, 512, n_class)
            self.block2 = Residual(512, 1024, n_class)
            self.block3 = Residual(1024, 512, n_class)
            self.block3_4 = Attention(512, 512)
            self.block4 = Linear(512, data_dim)

    def forward(self, input, label, dim, em_label):
        gen_input = torch.cat((input, label), -1)
        if dim <= 64:
            out1 = self.block1(gen_input, em_label)
            out2 = self.block2(out1, em_label)
            out3 = self.block3(out2, em_label)
            out4 = self.block4(out3, em_label)
            out5 = self.block5(out4, em_label)
            out6 = self.block6(out3 + out5, em_label)
            out7 = self.block7(out2 + out6, em_label)
            out7_8 = self.block7_8(out7)
            out8 = self.block8(out1 + out7_8)
            re = out8
        elif dim <= 128:
            out1 = self.block1(gen_input, em_label)
            out2 = self.block2(out1, em_label)
            out3 = self.block3(out2, em_label)
            out4 = self.block4(out3, em_label)
            out5 = self.block5(out2 + out4, em_label)
            out5_6 = self.block5_6(out5)
            out6 = self.block6(out1 + out5_6)
            re = out6
        else:
            out1 = self.block1(gen_input, em_label)
            out2 = self.block2(out1, em_label)
            out3 = self.block3(out2, em_label)
            out3_4 = self.block3_4(out3)
            out4 = self.block4(out1 + out3_4)
            re = out4

        return re


class Classify(nn.Module):
    def __init__(self, input_dim, target_dim):
        super(Classify, self).__init__()
        self.target_dim = target_dim
        dim = input_dim
        seq = []
        if input_dim >= 256:
            seq += [Attention(dim, dim)]
            seq += [Block(dim, 256)]
            seq += [Block(256, 128)]
            seq += [Block(128, 64)]
        elif input_dim >= 128:
            seq += [Attention(dim, dim)]
            seq += [Block(dim, 128)]
            seq += [Block(128, 64)]
        else:
            seq += [Attention(dim, dim)]
            seq += [Block(dim, 64)]

        dim = 64

        if self.target_dim >= 16:
            seq += [Linear(dim, 32)]
            dim = 32
        elif self.target_dim >= 8:
            seq += [Block(dim, 32)]
            seq += [Linear(32, 16)]
            dim = 16
        else:
            seq += [Block(dim, 32)]
            seq += [Block(32, 16)]
            seq += [Linear(16, 8)]
            dim = 8

        seq += [Linear(dim, self.target_dim)]
        self.seq = Sequential(*seq)

    def forward(self, input):
        output = self.seq(input)
        return output
