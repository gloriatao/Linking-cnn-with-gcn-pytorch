import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.layer_GCN import gcn_layer

def same_padding_3d(images, ksizes, strides=(1,1,1), rates=(1,1,1)):
    assert len(images.size()) == 5
    batch_size, channel, depth, rows, cols = images.size()

    out_depth = (depth + strides[0] - 1) // strides[0]
    out_rows = (rows + strides[1] - 1) // strides[1]
    out_cols = (cols + strides[2] - 1) // strides[2]

    effective_k_depth = (ksizes[0] - 1) * rates[0] + 1
    effective_k_row = (ksizes[1] - 1) * rates[1] + 1
    effective_k_col = (ksizes[2] - 1) * rates[2] + 1

    padding_depth = max(0, (out_depth - 1) * strides[0] + effective_k_depth - depth)
    padding_rows = max(0, (out_rows-1)*strides[1]+effective_k_row-rows)
    padding_cols = max(0, (out_cols-1)*strides[2]+effective_k_col-cols)

    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_front = int(padding_depth / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    padding_back = padding_depth - padding_front
    paddings = nn.ConstantPad3d((padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back),0) #Output: (N,C,Dout,Hout,Wout)(N, C, D_{out}, H_{out}, W_{out})(N,C,Dout​,Hout​,Wout​) where
    images = paddings(images)
    return images

class ConvLayer_BN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvLayer_BN, self).__init__()

        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=1, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        y = self.leakyrelu(self.bn(self.conv3d(x)))
        return y

class phi_fun(nn.Module):
    def __init__(self, cnnOFeat_len, input_channel=1, droupout_rate=0.5):
        super(phi_fun, self).__init__()

        # Initial convolution layers
        self.Conv_1 = ConvLayer_BN(input_channel, 32, kernel_size=(5,7,7), stride=1, padding=0)
        self.Conv_2 = ConvLayer_BN(32, 64, kernel_size=(5,7,7), stride=1, padding=0)
        self.Mp_3 = nn.MaxPool3d((2, 2, 2))
        self.Dp = nn.Dropout3d(droupout_rate)
        self.Conv_5 = ConvLayer_BN(64, 128, kernel_size=(2,5,5), stride=1, padding=0)
        # self.Dp_6 = nn.Dropout3d(droupout_rate)
        self.Fc_7 = nn.Linear(65536, 50)
        self.Fc_8 = nn.Linear(50, 100)
        self.Fc_9 = nn.Linear(100, cnnOFeat_len)
        # self.Dp_10 = nn.Dropout3d(droupout_rate)

    def forward(self, x):
        # bs = x.shape[0]
        # nodes = x.shape[1]
        # # reshape to support batch size > 1
        # x = x.reshape(bs*nodes, x.shape[2], x.shape[3], x.shape[4], x.shape[5])

        out = same_padding_3d(x, (5,7,7))
        out = self.Conv_1(out)
        out = same_padding_3d(out, (5,7,7))
        out = self.Conv_2(out)
        out = self.Mp_3(out)
        out = self.Dp(out)
        out = same_padding_3d(out, (2,5,5))
        out = self.Conv_5(out)
        out = self.Dp(out)

        out = out.view(out.shape[0], -1)
        out = self.Dp(out)
        out = self.Fc_7(out)
        out = self.Fc_8(out)
        out = self.Fc_9(out)
        out = self.Dp(out)

        # reshape to original batch size
        # out = out.reshape(bs, nodes, -1)
        return out # 128 10

class Av_CNN3D_model(nn.Module):
    def __init__(self, patch_sz, number_class, droupout_rate=0.5):
        super(Av_CNN3D_model, self).__init__()

        self.Phi_fun = phi_fun(input_channel=1, droupout_rate=droupout_rate) # input is single channel image with depth = 5
        self.Fc_11 = nn.Linear(1000, number_class)
        self.seen = 0

    def load_weights(self, weightfile):
        params = torch.load(weightfile, map_location=torch.device('cpu'))
        if 'seen' in params.keys():
            self.seen = params['seen']
            del params['seen']
        else:
            self.seen = 0
        self.load_state_dict(params['state_dict'])
        print('Load Weights from %s... Done!!!' % weightfile)
        del params

    def forward(self, x):
        out = self.Phi_fun(x)
        out = self.Fc_11(out)
        out = F.softmax(out, dim=1)
        return out

class Av_CNN_GCN_model(nn.Module):
    def __init__(self, cnnOFeat_len, gcnOFeat_len, gcnNumGaussian, gaussian_hidden_feat, number_neighbors=2, droupout_rate=0.5):
        super(Av_CNN_GCN_model, self).__init__()

        self.num_neighbors = number_neighbors
        self.Phi_fun = phi_fun(cnnOFeat_len=cnnOFeat_len, input_channel=1, droupout_rate=droupout_rate)
        self.gcn_layer = gcn_layer(num_Gaussian=gcnNumGaussian, gaussian_hidden_feat=gaussian_hidden_feat,
                                   IFeat_len=cnnOFeat_len, OFeat_len=gcnOFeat_len, lamda=1.0)             # num_gaussian >= num_neighbours
        self.seen = 0

    def load_weights(self, weightfile):
        params = torch.load(weightfile, map_location=torch.device('cpu'))
        if 'seen' in params.keys():
            self.seen = params['seen']
            del params['seen']
        else:
            self.seen = 0
        self.load_state_dict(params['state_dict'])
        print('Load Weights from %s... Done!!!' % weightfile)
        del params

    def forward(self, X_batch, NX_batch):
        X = self.Phi_fun(X_batch) # cnn out: batchsize, num of nodes, feature length

        NX = []
        for i in range(NX_batch.shape[1]):
            NX_batch_i = NX_batch[:,i,:,:,:]
            # NX_batch_i = NX_batch_i[:,None,:,:,:]
            tmp = self.Phi_fun(NX_batch_i)
            tmp = tmp[:, None, :]
            NX.append(tmp)  # the size of tmp [b, F] & NX is a list with n [b, F]
        NX = torch.cat(NX, axis=1)
        NX = NX.detach()  # cnn out: batchsize, num of nodes, num_neighbours, feature length

        # # reshape for gcn
        # bs = X.shape[0]
        # nodes = X.shape[1]
        # X = X.reshape(bs * nodes, -1)
        # NX = NX.reshape(bs * nodes, self.num_neighbors, -1)

        out = self.gcn_layer(X, NX)
        out = F.softmax(out, dim=1)
        # out = out.reshape(bs, nodes, -1)
        return out



