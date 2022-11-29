import torch.nn as nn
import torch
class SGNet(nn.Module):
    def __init__(self, in_planes,out_planes):
        super(SGNet, self).__init__()
        self.f = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.g = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.sm = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(in_planes, out_planes, (1, 1))

    def forward(self, content, style):
        F = self.f(mean_variance_norm(content)) 
        G = self.g(mean_variance_norm(style)) 
        H = self.h(style) 
        b, c, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        b, c, h, w = G.size()
        G = G.view(b, -1, w * h) 
        S = torch.bmm(F, G) 
        S = self.sm(S)
        b, c, h, w = H.size() 
        H = H.view(b, -1, w * h) 
        O = torch.bmm(H, S.permute(0, 2, 1)) 
        b, c, h, w = style.size()
        O = torch.mean(O.view(-1,6, c, h, w)+style.view(-1,6, c, h, w), dim = 1)
        O = self.out_conv(O)

        return O

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat
