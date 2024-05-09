import numpy as np
from torch import nn as nn

import archs.flow_modules.Split
from archs.flow_modules import flow
from archs.flow_modules.Split import Split2d
from archs.flow_modules.FlowStep import FlowStep
from utils.common import opt_get
from archs.flow_modules.dequantization import uniform_dequantization

def f_conv2d_bias(in_channels, out_channels):
    def padding_same(kernel, stride):
        return [((k - 1) * s + 1) // 2 for k, s in zip(kernel, stride)]

    padding = padding_same([3, 3], [1, 1])
    assert padding == [1, 1], padding
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=[1,1], stride=1, padding=1,
                  bias=True))

class FlowUpsamplerNet(nn.Module):
    def __init__(self, image_shape, hidden_channels, K, L=None,
                 actnorm_scale=1.0,
                 flow_permutation=None,
                 flow_coupling="affine",
                 LU_decomposed=False, opt=None):
        """
                             K                                      K
        --> [Squeeze] -> [FlowStep] -------------> [Squeeze] -> [FlowStep]
               ^                           v
               |          (L - 1)          |
               + --------------------------+
        """
        super().__init__()

        self.layers = nn.ModuleList()
        self.output_shapes = []
        self.L = opt_get(opt, ['flow', 'L'])
        self.K = opt_get(opt, ['flow', 'K'])
        if isinstance(self.K, int):
            self.K = [K for K in [K, ] * (self.L + 1)]

        self.opt = opt
        H, W, self.C = image_shape
        self.check_image_shape()
        assert H == W
        self.image_size = H

        opt_dequant = opt_get(self.opt, ['flow', 'dequantization'])
        if opt_dequant == None or opt_dequant['type'] == 'uniform':
            print('using uniform dequantization')
            self.dequant_flows = uniform_dequantization
        else: 
            raise NotImplementedError()

        self.levelToName = {
            0: 'fea_down1',
            1: 'fea_down2',
            2: 'fea_down4',
            3: 'fea_down8',
            4: 'fea_down16',
        }

        flow_permutation = self.get_flow_permutation(flow_permutation, opt)
        self.arch_rgbAffine(H, LU_decomposed, W, actnorm_scale, hidden_channels, opt)
        H, W = self.arch_upsampleAndSqueeze(H, W, opt)
        self.layers.append(
            FlowStep(in_channels=self.C,
                        hidden_channels=hidden_channels,
                        actnorm_scale=actnorm_scale,
                        flow_permutation=flow_permutation,
                        flow_coupling='noCoupling',
                        LU_decomposed=LU_decomposed, opt=opt,
                        normOpt={'type':'noNorm','position':None}))
        self.output_shapes.append(
            [-1, self.C, H, W])
        self.arch_additionalFlowAffine(H, LU_decomposed, W, actnorm_scale, hidden_channels, opt)

        normOpt = opt_get(opt, ['flow', 'norm'])
        injector_type = opt_get(opt, ['flow', 'pre_injector_type'])
        if injector_type:
            self.arch_FlowStep(H, 1, LU_decomposed, W, actnorm_scale, f'CondFeaAffine{injector_type}',
                flow_permutation,
                hidden_channels, normOpt, opt,)

        # Upsampler
        for level in range(1, self.L + 1):
            # 1. Squeeze
            H, W = self.arch_squeeze(H, W)

            # 2. K FlowStep
            for _ in range(self.K[level]):
                self.arch_additionalFlowAffine(H, LU_decomposed, W, actnorm_scale, hidden_channels, opt)
                
                for coup_type in opt_get(opt, ['flow', 'layer_type']):
                    coup_type1, coup_type2 = coup_type.split('_')
                    if 'CFA' in coup_type1:
                        self.arch_FlowStep(H, 1, LU_decomposed, W, actnorm_scale, f'CondFeaAffine{coup_type2}',
                                        flow_permutation, hidden_channels, normOpt, opt)
                    elif 'CSA' in coup_type1:
                        self.arch_FlowStep(H, 1, LU_decomposed, W, actnorm_scale, f'CondSelfAffine{coup_type2}',
                                        flow_permutation, hidden_channels, normOpt, opt)
                    else:
                        self.arch_FlowStep(H, 1, LU_decomposed, W, actnorm_scale, f'CondFea{coup_type1}AndCondSelf{coup_type2}',
                                        flow_permutation, hidden_channels, normOpt, opt)

            # Split
            self.arch_split(H, W, level, self.L, opt)

        affineInCh = opt_get(opt, ['flow', 'hidden_channels'])
        if opt_get(opt, ['flow', 'split', 'enable']):
            self.f = f_conv2d_bias(affineInCh, 2 * 3 * 64 // 2 // 2)
        else:
            self.f = f_conv2d_bias(affineInCh, 2 * 3 * 64)

        self.H = H
        self.W = W
        self.scaleH = self.image_size / H
        self.scaleW = self.image_size / W

    def arch_FlowStep(self, H, K, LU_decomposed, W, actnorm_scale, flow_coupling, flow_permutation,
                      hidden_channels, normOpt, opt):
        for k in range(K):
            position_name = self.get_position_name(H, self.opt['scale'])
            if normOpt: normOpt['position'] = position_name

            self.layers.append(
                FlowStep(in_channels=self.C,
                         hidden_channels=hidden_channels,
                         actnorm_scale=actnorm_scale,
                         flow_permutation=flow_permutation,
                         flow_coupling=flow_coupling,
                         acOpt=None,
                         position=position_name,
                         LU_decomposed=LU_decomposed, opt=opt, idx=k, normOpt=normOpt))
            self.output_shapes.append([-1, self.C, H, W])

    def arch_preFlow(self, K, LU_decomposed, actnorm_scale, hidden_channels, opt):
        self.preFlow = nn.ModuleList()
        preFlow = opt_get(opt, ['flow', 'preFlow'])
        flow_permutation = self.get_flow_permutation(None, opt)
        if preFlow:
            for k in range(K):
                self.preFlow.append(
                    FlowStep(in_channels=self.C,
                             hidden_channels=hidden_channels,
                             actnorm_scale=actnorm_scale,
                             flow_permutation=flow_permutation,
                             flow_coupling='noCoupling',
                             LU_decomposed=LU_decomposed, opt=opt))

    def arch_split(self, H, W, L, levels, opt):
        correct_splits = opt_get(opt, ['flow', 'split', 'correct_splits'], False)
        correction = 0 if correct_splits else 1
        if opt_get(opt, ['flow', 'split', 'enable']) and L < levels - correction:
            logs_eps = opt_get(opt, ['flow', 'split', 'logs_eps']) or 0
            consume_ratio = opt_get(opt, ['flow', 'split', 'consume_ratio']) or 0.5
            position_name = self.get_position_name(H, self.opt['scale'])
            position = position_name if opt_get(opt, ['flow', 'split', 'conditional']) else None
            cond_channels = opt_get(opt, ['flow', 'split', 'cond_channels'])
            cond_channels = 0 if cond_channels is None else cond_channels

            t = opt_get(opt, ['flow', 'split', 'type'], 'Split2d')

            if t == 'Split2d':
                split = archs.flow_modules.Split.Split2d(num_channels=self.C, logs_eps=logs_eps, position=position,
                                                     cond_channels=cond_channels, consume_ratio=consume_ratio, opt=opt)
            elif t == 'FlowSplit2d':
                split = archs.flow_modules.Split.FlowSplit2d(num_channels=self.C, logs_eps=logs_eps, position=position,
                                                         cond_channels=cond_channels, consume_ratio=consume_ratio,
                                                         opt=opt)

            self.layers.append(split)
            self.output_shapes.append([-1, split.num_channels_pass, H, W])
            self.C = split.num_channels_pass

    def arch_additionalFlowAffine(self, H, LU_decomposed, W, actnorm_scale, hidden_channels, opt):
        if 'additionalFlowNoAffine' in opt['flow']:
            n_additionalFlowNoAffine = int(opt['flow']['additionalFlowNoAffine'])
            flow_permutation = self.get_flow_permutation(None, opt)

            for _ in range(n_additionalFlowNoAffine):
                self.layers.append(
                    FlowStep(in_channels=self.C,
                             hidden_channels=hidden_channels,
                             actnorm_scale=actnorm_scale,
                             flow_permutation=flow_permutation,
                             flow_coupling='noCoupling',
                             LU_decomposed=LU_decomposed, opt=opt))
                self.output_shapes.append(
                    [-1, self.C, H, W])

    def arch_upsampleAndSqueeze(self, H, W, opt):
        if not 'UpsampleAndSqueeze' in opt['flow']:
            return H, W

        self.C = self.C * 2 * 2
        self.layers.append(flow.UpsampleAndSqueezeLayer())
        self.output_shapes.append([-1, self.C, H, W])
        return H, W

    def arch_squeeze(self, H, W):
        self.C, H, W = self.C * 4, H // 2, W // 2
        self.layers.append(flow.SqueezeLayer(factor=2))
        self.output_shapes.append([-1, self.C, H, W])
        return H, W

    def arch_rgbAffine(self, H, LU_decomposed, W, actnorm_scale, hidden_channels, opt):
        rgbAffine = opt_get(opt, ['flow', 'rgbAffine'])
        if rgbAffine is not None:
            for _ in range(rgbAffine['n_steps']):
                self.layers.append(
                    FlowStep(in_channels=self.C,
                             hidden_channels=hidden_channels,
                             actnorm_scale=actnorm_scale,
                             flow_permutation='invconv',
                             flow_coupling='affineCustom',
                             LU_decomposed=LU_decomposed, opt=opt,
                             acOpt=rgbAffine))
                self.output_shapes.append(
                    [-1, self.C, H, W])

    def get_flow_permutation(self, flow_permutation, opt):
        flow_permutation = opt['flow'].get('flow_permutation', 'invconv')
        return flow_permutation

    def check_image_shape(self):
        assert self.C == 1 or self.C == 3, ("image_shape should be HWC, like (64, 64, 3)"
                                            "self.C == 1 or self.C == 3")

    def forward(self, gt=None, condResults=None, z=None, epses=None, logdet=0., reverse=False, eps_std=None, label=None):
        if reverse:
            epses_copy = [eps for eps in epses] if isinstance(epses, list) else epses

            img, logdet = self.decode(condResults, z, eps_std, epses=epses_copy, logdet=logdet, label=label)
            return img, logdet
        else:
            assert gt is not None
            assert condResults is not None
            z, logdet = self.encode(gt, condResults, logdet=logdet, epses=epses, label=label)

            return z, logdet

    def encode(self, gt, condResults, logdet=0.0, epses=None, label=None):
        gt, logdet = self.dequant_flows(gt, logdet)

        fl_fea = gt
        reverse = False
        level_conditionals = {}

        L = opt_get(self.opt, ['flow', 'L'])

        for layer, shape in zip(self.layers, self.output_shapes):
            size = shape[2]
            level = int(np.log(self.image_size / size) / np.log(2))

            level_conditionals[level] = condResults[self.levelToName[level]]

            if isinstance(layer, FlowStep):
                fl_fea, logdet = layer(fl_fea, logdet, reverse=reverse, condResults=level_conditionals[level])
            elif isinstance(layer, Split2d):
                fl_fea, logdet = self.forward_split2d(epses, fl_fea, layer, logdet, reverse, level_conditionals[level],label=label)
            else:
                fl_fea, logdet = layer(fl_fea, logdet, reverse=reverse)

        z = fl_fea

        if not isinstance(epses, list):
            return z, logdet

        epses.append(z)
        return epses, logdet

    def forward_split2d(self, epses, fl_fea, layer, logdet, reverse, condResults, label=None):
        ft = None if layer.position is None else condResults[layer.position]
        fl_fea, logdet, eps = layer(fl_fea, logdet, reverse=reverse, eps=epses, ft=ft, label=label)

        if isinstance(epses, list):
            epses.append(eps)
        return fl_fea, logdet

    def decode(self, condResults, z, eps_std=None, epses=None, logdet=0.0, label=None):
        reverse = True
        z = epses.pop() if isinstance(epses, list) else z

        fl_fea = z
        level_conditionals = {}
        if not opt_get(self.opt, ['flow', 'levelConditional', 'conditional']) == True:
            for level in range(self.L + 1):
                level_conditionals[level] = condResults[self.levelToName[level]]

        for layer, shape in zip(reversed(self.layers), reversed(self.output_shapes)):
            size = shape[2]
            level = int(np.log(self.image_size / size) / np.log(2))

            if isinstance(layer, (Split2d)):
                fl_fea, logdet = self.forward_split2d_reverse(eps_std, epses, fl_fea, layer,
                                                              condResults[self.levelToName[level]], logdet=logdet, label=label)
            elif isinstance(layer, FlowStep):
                fl_fea, logdet = layer(fl_fea, logdet=logdet, reverse=True, condResults=level_conditionals[level])
            else:
                fl_fea, logdet = layer(fl_fea, logdet=logdet, reverse=True)

        assert fl_fea.shape[1] == 3
        return fl_fea, logdet

    def forward_split2d_reverse(self, eps_std, epses, fl_fea, layer, condResults, logdet, label=None):
        ft = None if layer.position is None else condResults[layer.position]

        fl_fea, logdet = layer(fl_fea, logdet=logdet, reverse=True,
                               eps=epses.pop() if isinstance(epses, list) else None,
                               eps_std=eps_std, ft=ft, label=label)
        return fl_fea, logdet

    def get_position_name(self, H, scale):
        downscale_factor = self.image_size // H
        position_name = 'fea_up{}'.format(scale / downscale_factor)
        return position_name
