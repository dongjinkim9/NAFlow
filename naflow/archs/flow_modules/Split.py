import torch
from torch import nn as nn

from archs.flow_modules import thops
from archs.flow_modules.FlowStep import FlowStep
from archs.flow_modules.flow import Conv2dZeros, GaussianDiag
from archs.flow_modules import flow
from utils.common import opt_get

import numpy as np


class Split2d(nn.Module):
    def __init__(self, num_channels, logs_eps=0, cond_channels=0, position=None, consume_ratio=0.5, opt=None):
        super().__init__()

        self.num_channels_consume = int(round(num_channels * consume_ratio))
        self.num_channels_pass = num_channels - self.num_channels_consume

        self.conv = Conv2dZeros(in_channels=self.num_channels_pass + cond_channels,
                                out_channels=self.num_channels_consume * 2)
        self.logs_eps = opt_get(opt, ['network_G', 'flow', 'split', 'eps'],  logs_eps)
        self.position = position
        self.opt = opt
        # self.classes = opt_get(opt, ['network_G', 'flow', 'shift', 'classes'])
        self.classes = opt_get(opt, ['datasets', 'train', 'classes'])

        C = self.num_channels_consume

        # parameters to model the domain shift
        # domain X is N(0,1)
        # domain Y is N(mean_shift, cov_shift @ cov_shift^T)
        # self.I = torch.nn.Parameter(torch.eye(C, requires_grad=False), requires_grad=False)
        # self.mean_shifts = [torch.nn.Parameter(torch.zeros(C,requires_grad=True), requires_grad=True) for i in range(len(self.classes))]
        # std_init_shift = opt_get(self.opt, ['network_G', 'flow', 'shift', 'std_init_shift'], 1.0)
        # self.cov_shifts = [torch.nn.Parameter(torch.eye(C,requires_grad=True) * std_init_shift, requires_grad=True) for i in range(len(self.classes))]
        self.I = torch.nn.Parameter(torch.eye(C, requires_grad=False), requires_grad=False)
        
        trainable_mean = opt_get(self.opt, ['network_G', 'flow', 'shift', 'trainable_mean'])
        trainable_var = opt_get(self.opt, ['network_G', 'flow', 'shift', 'trainable_var'])
        self.mean_shifts = {
            c : torch.nn.Parameter(torch.zeros(C,requires_grad=trainable_mean), requires_grad=trainable_mean) 
            for c in self.classes}
        std_init_shift = opt_get(self.opt, ['network_G', 'flow', 'shift', 'std_init_shift'], 1.0)
        self.cov_shifts = {
            c : torch.nn.Parameter(torch.eye(C,requires_grad=trainable_var) * std_init_shift, requires_grad=trainable_var) 
            for c in self.classes}
        
        for c in self.classes:
            self.register_parameter(f"mean_shift_{c}", self.mean_shifts[c])
            self.register_parameter(f"cov_shift_{c}", self.cov_shifts[c])

    def split2d_prior(self, z, ft):
        if ft is not None:
            z = torch.cat([z, ft], dim=1)
        h = self.conv(z)
        return thops.split_feature(h, "cross")

    def exp_eps(self, logs):
        if opt_get(self.opt, ['network_G', 'flow', 'split', 'tanh'], False):
            return torch.exp(torch.tanh(logs)) + self.logs_eps

        return torch.exp(logs) + self.logs_eps

    def forward(self, input, logdet=0., reverse=False, eps_std=None, eps=None, ft=None, label=None):
        # domX = y_onehot == 0
        # domY = y_onehot == 1

        if type(label) != torch.Tensor:
            # label = torch.Tensor(label)
            label = np.array(label)
        assert len(label.shape) == 1, 'labels must be one dimensional'

        dom = {c : label == c for c in self.classes}
        
        if not reverse:
            self.input = input

            z1, z2 = self.split_ratio(input)

            self.z1 = z1
            self.z2 = z2

            mean, logs = self.split2d_prior(z1, ft)
            
            eps = (z2 - mean) / (self.exp_eps(logs) + 1e-6)

            assert eps.shape[0] == label.shape[0], 'need one class label per datapoint'

            cov_shifteds = dict()
            for name, cov_shifted in self.cov_shifts.items():
                cov_shifteds[name] = torch.matmul(self.I + cov_shifted, self.I + cov_shifted.T)
            mean_shifteds = self.mean_shifts

            ll = torch.zeros(eps.shape[0], device=eps.device)
            for k,v in dom.items():
                if k == 'GT':
                    ll[v] = flow.Gaussian.logp(None, None, eps[v])
                else:
                    ll[v] = flow.Gaussian.logp(mean=mean_shifteds[k], cov=cov_shifteds[k], x=eps[v])


            logdet = logdet + ll - logs.sum(dim=[1,2,3])

            return z1, logdet, eps
        else:
            z1 = input
            mean, logs = self.split2d_prior(z1, ft)

            if eps is None: # sample eps
                eps = GaussianDiag.sample_eps(mean.shape, eps_std)
                eps = eps.to(mean.device)
                
                shape = mean.shape

                for k,v in dom.items():
                    if k == 'GT': continue
                    domY = v
                    if domY.any(): # sample and add u
                        z_noise = GaussianDiag.sample_eps(shape, eps_std)[domY].to(mean.device)
                        eps[domY] = (eps[domY] + self.mean_shifts[k].reshape(1,self.mean_shifts[k].shape[0],1,1) 
                                    + torch.matmul(self.cov_shifts[k], z_noise.T).T)
            else:
                eps = eps.to(mean.device)

            assert eps.shape[0] == label.shape[0], 'need one class label per datapoint'

            cov_shifteds = dict()
            for name, cov_shifted in self.cov_shifts.items():
                cov_shifteds[name] = self.I + torch.matmul(cov_shifted, cov_shifted.T)
            mean_shifteds = self.mean_shifts

            ll = torch.zeros(eps.shape[0], device=eps.device)

            for k,v in dom.items():
                if k == 'GT':
                    ll[v] = flow.Gaussian.logp(None, None, eps[v])
                else:
                    ll[v] = flow.Gaussian.logp(mean=mean_shifteds[k], cov=cov_shifteds[k], x=eps[v])


            z2 = mean + self.exp_eps(logs) * eps

            z = thops.cat_feature(z1, z2)

            logdet = logdet - ll + logs.sum(dim=[1,2,3])

            return z, logdet

    def split_ratio(self, input):
        z1, z2 = input[:, :self.num_channels_pass, ...], input[:, self.num_channels_pass:, ...]
        return z1, z2