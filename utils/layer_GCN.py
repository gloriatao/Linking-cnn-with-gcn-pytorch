import torch
import torch.nn as nn
import torch.nn.functional as F

def bdot(a, b):
    B = a.shape[0]
    b = b[None,:,:]
    b = b.repeat(B,1,1)
    return torch.bmm(a, b)

class GCNlayer(nn.Module):
    "Define Graph convolution layer"
    def __init__(self, num_Gaussian = 6, gaussian_hidden_feat=3, IFeat_len=10, OFeat_len = 10, lamda = 0.1, use_bias = True):
        super(GCNlayer, self).__init__()

        self.filters = num_Gaussian
        self.IFeat_len = IFeat_len
        self.outFeat_len = OFeat_len
        self.gaussian_hidden_feat = gaussian_hidden_feat
        self.use_bias = use_bias
        self.lamda = lamda

        # Params for estimate gaussian kenel.
        self.mu = nn.Parameter(torch.zeros(size=(self.filters, self.gaussian_hidden_feat)))
        nn.init.xavier_uniform_(self.mu.data)
        self.sigma = nn.Parameter(torch.zeros(size=(self.filters, self.gaussian_hidden_feat)))
        nn.init.xavier_uniform_(self.sigma.data)

        # Params for gcn.
        self.Aweight = nn.Parameter(torch.zeros(size=(self.IFeat_len, self.gaussian_hidden_feat)))
        nn.init.xavier_uniform_(self.Aweight.data)
        self.theta = nn.Parameter(torch.zeros(size=(self.IFeat_len, self.outFeat_len)))
        nn.init.xavier_uniform_(self.theta.data)

        if self.use_bias:
            self.Abias = nn.Parameter(torch.zeros(size=(1, self.gaussian_hidden_feat)))
            self.bias = nn.Parameter(torch.zeros(size=(1, self.outFeat_len)))

    def weight_fun(self, X, Nx):
        # x with size [node_num*batchsz, feature_len]
        # Nx with size [node_num*batchsz, neigbours, feature_len]
        X = X[:, None, :]
        dif = X - Nx
        mu_x = bdot(dif, self.Aweight)
        if self.use_bias:  # mu_x with size [b, N, h], where h is the number of hidden feature
            mu_x += self.Abias
        mu_x = F.tanh(mu_x)

        mu_x = torch.sum(mu_x, dim=1)
        mu_x = mu_x[:,None,:]
        mu_x = mu_x.repeat(1,self.filters,1)

        dif_mu = torch.sum(-0.5 * torch.mul(mu_x - self.mu, mu_x - self.mu)/ (1e-14 + torch.mul(self.sigma, self.sigma)), dim=-1)
        weight = torch.exp( dif_mu )
        weight = weight / (1e-14 + torch.sum(weight, axis=-1, keepdims = True))

        return weight # ouput weight with size [b, Num of gaussians]

    def forward(self, X, Nx):
        # x with size [node_num*batchsz, feature_len]
        # Nx with size [node_num*batchsz, neigbours, feature_len]
        # x_out with size [node_num*batchsz, num_classes]

        weight = self.weight_fun(X, Nx) # with size of [b, N]
        weight = self.lamda * weight

        # # feature representation
        weight = weight[:,None,:]
        Nx = torch.sum(Nx, dim=1)
        Nx = Nx[:,None,:]
        Nx = Nx.repeat(1,self.filters,1)

        X_merge = torch.bmm(weight, Nx).squeeze() # with size [node_num*batchsz, feature_len]
        H = (X_merge + X) / (1 + torch.sum(weight, axis=-1))
        x_out = torch.mm(H, self.theta)  # feature transform
        if self.use_bias:
            x_out += self.bias

        x_out = F.relu(x_out)
        return x_out

def gcn_layer(num_Gaussian = 1,
              gaussian_hidden_feat=3,
              IFeat_len = 100,
              OFeat_len = 2,
              use_bias = True,
              lamda = 0.1):
    "Functional interface of gcn_layer"
    layer = GCNlayer(
        num_Gaussian = num_Gaussian,
        gaussian_hidden_feat = gaussian_hidden_feat,
        IFeat_len=IFeat_len,
        OFeat_len = OFeat_len,
        lamda=lamda,
        use_bias=use_bias)

    return layer

# if __name__ == "__main__":
#
#     X = torch.randn((128,100))
#     NX = torch.randn((128,2,100))
#     model = gcn_layer( Num_Gaussian=1, n_hidden_feat=10, OFeat_len=100)
#     out = model(X, NX)
#
#
#     print(X)