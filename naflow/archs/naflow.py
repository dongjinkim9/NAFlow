import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import archs.flow_modules.thops as thops
import archs.flow_modules.flow as flow
from archs.flow_modules.FlowUpsamplerNet import FlowUpsamplerNet
from utils.common import opt_get

class InjectorNet(nn.Module):
    def __init__(self, in_nc, nf, scale, opt):
        self.opt = opt
        super(InjectorNet, self).__init__()
        self.scale = scale
        
        self.pixel_down = nn.PixelUnshuffle(2)
        if self.scale >= 1:
            self.down1 = nn.Conv2d(in_nc, nf, 1, 1, 'same', bias=True)
        if self.scale >= 2:
            self.down2 = nn.Conv2d(nf, nf//4, 1, 1, 'same', bias=True)
        if self.scale >= 4:
            self.down4 = nn.Conv2d(nf, nf//4, 1, 1, 'same', bias=True)
        if self.scale >= 8:
            self.down8 = nn.Conv2d(nf, nf//4, 1, 1, 'same', bias=True)
        if self.scale >= 16:
            self.down16 = nn.Conv2d(nf, nf//4, 1, 1, 'same', bias=True)

    def forward(self, x):
        block_results = {}
        fea_down1, fea_down2, fea_down4, fea_down8, fea_down16 = None, None, None, None, None

        if self.scale >= 1:
            fea_down1 = self.down1(x)
        if self.scale >= 2:
            fea_down2 = self.down2(fea_down1)
            fea_down2 = self.pixel_down(fea_down2)
        if self.scale >= 4:
            fea_down4 = self.down4(fea_down2)
            fea_down4 = self.pixel_down(fea_down4)
        if self.scale >= 8:
            fea_down8 = self.down8(fea_down4)
            fea_down8 = self.pixel_down(fea_down8)
        if self.scale >= 16:
            fea_down16 = self.down16(fea_down8)
            fea_down16 = self.pixel_down(fea_down16)


        results = {'fea_down1': fea_down1,
                   'fea_down2': fea_down2,
                   'fea_down4': fea_down4,
                   'fea_down8': fea_down8,
                   'fea_down16': fea_down16,}


        for k, v in block_results.items():
            results[k] = v
        return results 


class NAFlow(nn.Module):
    def __init__(self, opt):
        super(NAFlow, self).__init__()

        self.opt = opt
        
        self.cond_net = InjectorNet(in_nc=opt['in_nc'], nf=opt['nf'], scale=opt['scale'], opt=opt)
        self.set_condnet_to_train = opt['train_cond_net'] is not None
        self.set_condnet_training(self.set_condnet_to_train)

        hidden_channels = opt_get(opt, ['flow', 'hidden_channels'])
        self.flowUpsamplerNet = FlowUpsamplerNet(
            image_shape=(opt['flow']['img_size'], opt['flow']['img_size'], 3), hidden_channels= hidden_channels, 
            K=opt['flow']['K'], flow_coupling=opt['flow']['coupling'],
            LU_decomposed=opt_get(opt, ['flow', 'LU_decomposed'], False), opt=opt)

        self.classes = opt_get(opt, ['classes'])
        # calculate C
        if opt_get(opt, ['flow', 'L']) == 0 or opt_get(opt, ['flow', 'split', 'enable']):
            C = self.flowUpsamplerNet.C
        else:
            L = opt_get(opt, ['flow', 'L']) or 3
            fac = 2 ** (L - 3)
            C = 3 * 8 * 8 * fac * fac
            C = int(C)

        self.I = torch.nn.Parameter(torch.eye(C, requires_grad=False), requires_grad=False)

        # Prior distributions : N(mean_shift, cov_shift @ cov_shift^T)
        trainable_mean = opt_get(opt, ['flow', 'shift', 'trainable_mean'])
        trainable_var = opt_get(opt, ['flow', 'shift', 'trainable_var'])
        self.mean_shifts = {
            c : torch.nn.Parameter(torch.zeros(C,requires_grad=trainable_mean) + 1e-6, requires_grad=trainable_mean) 
            for c in self.classes}

        self.std_type = opt_get(opt, ['flow', 'shift', 'std_type'], 'full')
        std_init_shift = opt_get(opt, ['flow', 'shift', 'std_init_shift'], 1.0)

        if self.std_type == 'full':
            self.cov_shifts = {c : torch.nn.Parameter(
                torch.eye(C,requires_grad=trainable_var)*std_init_shift, requires_grad=trainable_var) for c in self.classes}
            self.logp = flow.Gaussian.logp
        elif self.std_type == 'diagonal':
            self.cov_shifts = {c : torch.nn.Parameter(
                torch.ones(C,requires_grad=trainable_var)*std_init_shift, requires_grad=trainable_var) for c in self.classes}
            self.logp = flow.GaussianDiag.logp
        else:
            raise NotImplementedError()
        
        for c in self.classes:
            self.register_parameter(f"mean_shift_{c}", self.mean_shifts[c])
            self.register_parameter(f"cov_shift_{c}", self.cov_shifts[c])

    def set_condnet_training(self, trainable):
        for p in self.cond_net.parameters():
            p.requires_grad = trainable
        print(f"set conditional encoder training: {trainable}")

    def forward(self, input=None, cond=None, z=None, eps_std=None, 
                reverse=False, epses=None, cond_enc=None, label=None):
        if not reverse:
            return self.normal_flow(input, cond, epses=epses, lr_enc=cond_enc, label=label)
        else:
            return self.reverse_flow(cond, z, label=label, eps_std=eps_std, epses=epses, lr_enc=cond_enc)

    def normal_flow(self, gt, lr, label=None, epses=None, lr_enc=None):
        if lr_enc is None:
            lr_enc = self.cond_net(lr)

        logdet = torch.zeros_like(gt[:, 0, 0, 0])
        pixels = thops.pixels(gt)

        z = gt

        # Encode
        epses, logdet = self.flowUpsamplerNet(condResults=lr_enc, gt=z, logdet=logdet, reverse=False, epses=epses, label=label)
        
        objective = logdet.clone()
        
        if isinstance(epses, (list, tuple)):
            z = epses[-1]
        else:
            z = epses

        if type(label) != torch.Tensor:
            label = np.array(label)
        assert z.shape[0] == label.shape[0], 'need one class label per datapoint'
        assert len(label.shape) == 1, 'labels must be one dimensional'
        
        dom = {c : label == c for c in self.classes}

        mean_shifteds = self.mean_shifts
        cov_shifteds = dict()
        for name, cov_shifted in self.cov_shifts.items():
            if self.std_type == 'full':
                # https://en.wikipedia.org/wiki/Square_root_of_a_matrix#Positive_semidefinite_matrices
                cov_shifteds[name] = torch.matmul(cov_shifted, cov_shifted.T)
            elif self.std_type == 'diagonal':
                cov_shifteds[name] = cov_shifted.pow(2)

        ll = torch.zeros(z.shape[0], device=z.get_device() if z.get_device() >= 0 else None)

        for k,v in dom.items():
            if k == 'GT':
                ll[v] = self.logp(None, None, z[v])
            else:
                ll[v] = self.logp(mean_shifteds[k], cov_shifteds[k], z[v])

        objective = objective + ll

        nll = (-objective) / float(np.log(2.) * pixels)

        self.dom, self.mean_shifteds, self.cov_shifteds = dom, mean_shifteds, cov_shifteds
        self.z, self.label, self.pixels = z, label, pixels

        if isinstance(epses, list):
            return epses, nll, logdet
        return z, nll, logdet

    def reverse_flow(self, lr, z, label, eps_std, epses=None, lr_enc=None):
        # decode
        if z is None and epses is None: # sample z
            C = self.flowUpsamplerNet.C
            H = int(self.opt['scale'] * lr.shape[2] // self.flowUpsamplerNet.scaleH)
            W = int(self.opt['scale'] * lr.shape[3] // self.flowUpsamplerNet.scaleW)          

            batch_size = lr.shape[0]
            shape = (batch_size, C, H, W)

            z = flow.GaussianDiag.sample_eps(shape, eps_std).to(lr.device)

            if type(label) != torch.Tensor:
                label = np.array(label)
            
            assert z.shape[0] == label.shape[0], 'need one class label per datapoint'
            assert len(label.shape) == 1, 'labels must be one dimesntional'

            dom = {c : label == c for c in self.classes}
            assert any([v.any() for k,v in dom.items()])
            for k,v in dom.items():
                if k == 'GT': continue
                domY = v
                if domY.any(): # sample and add u
                    z_noise = flow.GaussianDiag.sample_eps(shape, eps_std)[domY].to(lr.device)
                    if self.std_type == 'full':
                        z[domY] = torch.einsum('cc,bchw->bchw',self.cov_shifts[k],z[domY]) + self.mean_shifts[k].reshape(1,self.mean_shifts[k].shape[0],1,1) 
                    elif self.std_type == 'diagonal':
                        z[domY] = torch.einsum('c,bchw->bchw',self.cov_shifts[k],z[domY]) + self.mean_shifts[k].reshape(1,self.mean_shifts[k].shape[0],1,1)

        # Setup
        logdet = torch.zeros_like(lr[:, 0, 0, 0])

        if lr_enc is None:
            lr_enc = self.cond_net(lr)

        x, logdet = self.flowUpsamplerNet(condResults=lr_enc, z=z, eps_std=eps_std, reverse=True, epses=epses,
                                          logdet=logdet, label=label)
        return x, logdet
    
    def noise_aware_sampling(self, z, class_threshold=None, temperature=1., verbose=False):
        assert z.shape[0] == 1, 'only support single batch'
        # if network does not have split operations
        if type(z) == torch.Tensor:       
            class_label = list(self.mean_shifts.keys())
            pixels = thops.pixels(z)

            if class_threshold == None:
                class_threshold = len(class_label)

            mean_shifteds = self.mean_shifts
            cov_shifteds = dict()
            for name, cov_shifted in self.cov_shifts.items():
                if self.std_type == 'full':
                    cov_shifteds[name] = torch.matmul(cov_shifted, cov_shifted.T)
                elif self.std_type == 'diagonal':
                    cov_shifteds[name] = cov_shifted.pow(2)

            lls_from_total_class = [] 
            for ml in class_label:
                ll = self.logp(mean_shifteds[ml], cov_shifteds[ml], z)
                lls_from_total_class.append(ll)
            lls_from_total_class = torch.stack(lls_from_total_class) # (num_classes, batch)

            pred_class = torch.argsort(lls_from_total_class, dim = 0, descending=True)
            class_label = np.array(class_label)
            filtered_ll = lls_from_total_class[pred_class[:class_threshold].squeeze(-1)] / (pixels*temperature)
            softmax_prob = torch.softmax(filtered_ll, dim=0).squeeze(-1)
            label = class_label[pred_class[:class_threshold].cpu()].reshape(-1).tolist()      
          
            new_mean = torch.sum(torch.stack([self.mean_shifts[l]*sp for l,sp in zip(label,softmax_prob)]),dim=0)
            new_std = torch.sum(torch.stack([self.cov_shifts[l]*sp for l,sp in zip(label,softmax_prob)]),dim=0)
            
            new_z = flow.GaussianDiag.sample_eps(z.shape, 1.0).to(z.device)
            z_noise = flow.GaussianDiag.sample_eps(z.shape, 1.0).to(z.device)
            if self.std_type == 'full':
                new_z = torch.einsum('cc,bchw->bchw',new_std,new_z) + new_mean.reshape(1,new_mean.shape[0],1,1) 
            elif self.std_type == 'diagonal':
                new_z = torch.einsum('c,bchw->bchw',new_std,new_z) + new_mean.reshape(1,new_mean.shape[0],1,1)
                     
            if verbose:
                print(f"{label=}") 
                print(f"{softmax_prob=}")
            return new_z

        # if network has split operations
        elif type(z) == list:
            raise NotImplementedError()
