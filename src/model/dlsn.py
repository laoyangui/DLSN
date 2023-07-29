from model import common
import torch.nn as nn
import torch
from model import attention
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def make_model(args, parent=False):
    return DLSN(args)

## Local Features Fusion Block (LFFB)
class LFFB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(LFFB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Global Learnable Attention-based Features Fusion Module (GLAFFM)
class GLAFFM(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks, chunk_size, n_hashes):
        super(GLAFFM, self).__init__()
        modules_body = []
        modules_body = [
            LFFB(conv, n_feat, kernel_size, 16, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)] # To stabilize the training, you can try lowering the value of res_scale.

        modules_body.append(attention.GLA(channels=n_feat, chunk_size=chunk_size, n_hashes=n_hashes, reduction=reduction, res_scale=1)) # To stabilize the training, you can try lowering the value of res_scale.
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Deep Learnable Similarity Network (DLSN)
class DLSN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(DLSN, self).__init__()
        self.a = nn.Parameter(torch.Tensor([0]))
        self.a.requires_grad=True
        n_modules = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction 
        scale = args.scale[0]
        act = nn.ReLU(True)
        
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        
        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = []
        for i in range(n_modules):
            m_body.append(GLAFFM(conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks, chunk_size=args.chunk_size, n_hashes=args.n_hashes))

        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x 

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('msa') or name.find('a') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('msa') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
