import torch
from torch import nn as nn

from archs.flow_modules import thops
from archs.flow_modules.flow import Conv2d, Conv2dZeros
from utils.common import opt_get

class AffineCoupling(nn.Module):
    def __init__(self, acOpt):
        super().__init__()
        self.in_channels = acOpt['in_channels']
        self.kernel_hidden = acOpt['kernel_hidden']
        self.affine_eps = acOpt['affine_eps']
        self.n_hidden_layers = acOpt['n_hidden_layers']
        self.channels_for_nn = acOpt['channels_for_nn']
        self.channels_for_co = acOpt['in_channels'] - acOpt['channels_for_nn']
        self.hidden_channels = acOpt['hidden_channels']

        if self.channels_for_nn is None:
            self.channels_for_nn = self.in_channels // 2

        self.f = self.F(in_channels=self.channels_for_nn,
                        out_channels=self.channels_for_co * 2,
                        hidden_channels=self.hidden_channels,
                        kernel_hidden=self.kernel_hidden,
                        n_hidden_layers=self.n_hidden_layers)

    def forward(self, input: torch.Tensor, logdet=None, reverse=False):
        if not reverse:
            z = input
            assert z.shape[1] == self.in_channels, (z.shape[1] == self.in_channels)
            z1, z2 = self.split(z)
            scale, shift = self.feature_extract(z1)
            assert z1.shape[1] == self.channels_for_nn
            assert z2.shape[1] == self.channels_for_co
            assert scale.shape[1] == shift.shape[1], (scale.shape[1], shift.shape[1])
            assert scale.shape[1] == z2.shape[1], (scale.shape[1], z1.shape[1], z2.shape[1])

            z2 = z2 + shift
            z2 = z2 * scale

            logdet = self.get_logdet(logdet, scale)
            z = thops.cat_feature(z1, z2)
            output = z
        else:
            z = input
            z1, z2 = self.split(z)
            scale, shift = self.feature_extract(z1)

            z2 = z2 / scale
            z2 = z2 - shift

            logdet = -self.get_logdet(logdet, scale)
            z = thops.cat_feature(z1, z2)
            output = z
        return output, logdet

    def get_logdet(self, logdet, scale):
        logdet = thops.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        return logdet

    def feature_extract(self, z1):
        h = self.f(z1)
        shift, scale = thops.split_feature(h, "cross")
        scale = (torch.sigmoid(scale + 2.) + self.affine_eps)
        return scale, shift

    def split(self, z):
        z1 = z[:, :self.channels_for_nn]
        z2 = z[:, self.channels_for_nn:]
        assert z1.shape[1] + z2.shape[1] == z.shape[1], (z1.shape[1], z2.shape[1], z.shape[1])
        return z1, z2

    def F(self, in_channels, out_channels, hidden_channels, kernel_hidden=1, n_hidden_layers=1):
        layers = [Conv2d(in_channels, hidden_channels), nn.ReLU(inplace=False)]

        for _ in range(n_hidden_layers):
            layers.append(Conv2d(hidden_channels, hidden_channels, kernel_size=[kernel_hidden, kernel_hidden]))
            layers.append(nn.ReLU(inplace=False))
        layers.append(Conv2dZeros(hidden_channels, out_channels))

        return nn.Sequential(*layers)


class CondAffineCoupling(nn.Module):
    def __init__(self, acOpt, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.in_channels_rrdb = acOpt['in_channels_rrdb']
        self.kernel_hidden = acOpt['kernel_hidden']
        self.affine_eps = acOpt['affine_eps']
        self.n_hidden_layers = acOpt['n_hidden_layers']
        self.channels_for_nn = acOpt['channels_for_nn']

        if self.channels_for_nn is None:
            self.channels_for_nn = self.in_channels // 2
        self.channels_for_co = self.in_channels - self.channels_for_nn

        self.hidden_channels = acOpt['hidden_channels']

        if self.channels_for_nn is None:
            self.channels_for_nn = self.in_channels // 2

        self.f = self.F(in_channels=self.channels_for_nn + self.in_channels_rrdb,
                        out_channels=self.channels_for_co * 2,
                        hidden_channels=self.hidden_channels,
                        kernel_hidden=self.kernel_hidden,
                        n_hidden_layers=self.n_hidden_layers)

    def forward(self, input: torch.Tensor, logdet=None, reverse=False, ft=None):
        if not reverse:
            z = input
            assert z.shape[1] == self.in_channels, (z.shape[1], self.in_channels)
            z1, z2 = self.split(z)
            #print(z1.shape, ft.shape, self.channels_for_nn, self.in_channels_rrdb)
            scale, shift = self.feature_extract(z1, ft)
            assert z1.shape[1] == self.channels_for_nn, (z1.shape[1], self.channels_for_nn)
            assert z2.shape[1] == self.channels_for_co, (z2.shape[1], self.channels_for_co)
            assert scale.shape[1] == shift.shape[1], (scale.shape[1], shift.shape[1])
            assert scale.shape[1] == z2.shape[1], (scale.shape[1], z1.shape[1], z2.shape[1])

            z2 = z2 + shift
            z2 = z2 * scale

            logdet = self.get_logdet(logdet, scale)
            z = thops.cat_feature(z1, z2)
            output = z
        else:
            z = input
            z1, z2 = self.split(z)
            scale, shift = self.feature_extract(z1, ft)

            z2 = z2 / scale
            z2 = z2 - shift

            logdet = -self.get_logdet(logdet, scale)
            z = thops.cat_feature(z1, z2)
            output = z
        return output, logdet

    def get_logdet(self, logdet, scale):
        logdet = thops.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        return logdet

    def feature_extract(self, z1, ft):
        z = torch.cat([z1, ft], dim=1)
        h = self.f(z)
        shift, scale = thops.split_feature(h, "cross")
        scale = (torch.sigmoid(scale + 2.) + self.affine_eps)
        return scale, shift

    def split(self, z):
        z1 = z[:, :self.channels_for_nn]
        z2 = z[:, self.channels_for_nn:]
        assert z1.shape[1] + z2.shape[1] == z.shape[1], (z1.shape[1], z2.shape[1], z.shape[1])
        return z1, z2

    def F(self, in_channels, out_channels, hidden_channels, kernel_hidden=1, n_hidden_layers=1):
        layers = [Conv2d(in_channels, hidden_channels), nn.ReLU(inplace=False)]

        for _ in range(n_hidden_layers):
            layers.append(Conv2d(hidden_channels, hidden_channels, kernel_size=[kernel_hidden, kernel_hidden]))
            layers.append(nn.ReLU(inplace=False))
        layers.append(Conv2dZeros(hidden_channels, out_channels))

        return nn.Sequential(*layers)


class CondNormAffineCoupling(nn.Module):
    def __init__(self, acOpt, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.in_channels_rrdb = acOpt['in_channels_rrdb']
        self.kernel_hidden = acOpt['kernel_hidden']
        self.affine_eps = acOpt['affine_eps']
        self.n_hidden_layers = acOpt['n_hidden_layers']
        self.channels_for_nn = acOpt['channels_for_nn']

        if self.channels_for_nn is None:
            self.channels_for_nn = self.in_channels // 2
        self.channels_for_co = self.in_channels - self.channels_for_nn

        self.hidden_channels = acOpt['hidden_channels']

        if self.channels_for_nn is None:
            self.channels_for_nn = self.in_channels // 2

        self.fNorm = self.F(in_channels=self.in_channels_rrdb,
                            out_channels=self.channels_for_nn * 2,
                            hidden_channels=self.in_channels_rrdb, kernel_hidden=1, n_hidden_layers=1)
        self.f = self.F(in_channels=self.channels_for_nn,
                        out_channels=self.channels_for_co * 2,
                        hidden_channels=self.hidden_channels,
                        kernel_hidden=self.kernel_hidden,
                        n_hidden_layers=self.n_hidden_layers)

    def forward(self, input: torch.Tensor, logdet=None, reverse=False, ft=None):
        if not reverse:
            z = input
            assert z.shape[1] == self.in_channels, (z.shape[1], self.in_channels)
            z1, z2 = self.split(z)
            scale, shift = self.feature_extract(z1, ft)
            assert z1.shape[1] == self.channels_for_nn, (z1.shape[1], self.channels_for_nn)
            assert z2.shape[1] == self.channels_for_co, (z2.shape[1], self.channels_for_co)
            assert scale.shape[1] == shift.shape[1], (scale.shape[1], shift.shape[1])
            assert scale.shape[1] == z2.shape[1], (scale.shape[1], z1.shape[1], z2.shape[1])

            z2 = z2 + shift
            z2 = z2 * scale

            logdet = self.get_logdet(logdet, scale)
            z = thops.cat_feature(z1, z2)
            output = z
        else:
            z = input
            z1, z2 = self.split(z)
            scale, shift = self.feature_extract(z1, ft)

            z2 = z2 / scale
            z2 = z2 - shift

            logdet = -self.get_logdet(logdet, scale)
            z = thops.cat_feature(z1, z2)
            output = z
        return output, logdet

    def get_logdet(self, logdet, scale):
        logdet = thops.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        return logdet

    def feature_extract(self, z1, ft):
        hFt = self.fNorm(ft)
        shiftFt, scaleFt = thops.split_feature(hFt, "cross")
        scaleFt = (torch.sigmoid(scaleFt + 2.) + self.affine_eps)
        z1 = z1 + shiftFt
        z1 = z1 * scaleFt  # Always forward
        h = self.f(z1)
        shift, scale = thops.split_feature(h, "cross")
        scale = (torch.sigmoid(scale + 2.) + self.affine_eps)
        return scale, shift

    def split(self, z):
        z1 = z[:, :self.channels_for_nn]
        z2 = z[:, self.channels_for_nn:]
        assert z1.shape[1] + z2.shape[1] == z.shape[1], (z1.shape[1], z2.shape[1], z.shape[1])
        return z1, z2

    def F(self, in_channels, out_channels, hidden_channels, kernel_hidden=1, n_hidden_layers=1):
        layers = [Conv2d(in_channels, hidden_channels), nn.ReLU(inplace=False)]

        for _ in range(n_hidden_layers):
            layers.append(Conv2d(hidden_channels, hidden_channels, kernel_size=[kernel_hidden, kernel_hidden]))
            layers.append(nn.ReLU(inplace=False))
        layers.append(Conv2dZeros(hidden_channels, out_channels))

        return nn.Sequential(*layers)


class CondFtAffineCoupling(nn.Module):
    def __init__(self, acOpt, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.in_channels_rrdb = acOpt['in_channels_rrdb']
        self.kernel_hidden = acOpt['kernel_hidden']
        self.n_hidden_layers = acOpt['n_hidden_layers'] if 'n_hidden_layers' in acOpt.keys() else 1
        self.hidden_channels = acOpt['hidden_channels']
        self.affine_eps = acOpt['affine_eps']
        self.fBypassToFtNorm_hidden_layers = acOpt['fBypassToFtNorm_hidden_layers'] \
            if 'fBypassToFtNorm_hidden_layers' in acOpt.keys() else self.n_hidden_layers
        self.fFtPreprocess_hidden_layers = acOpt['fFtPreprocess_hidden_layers'] \
            if 'fFtPreprocess_hidden_layers' in acOpt.keys() else self.n_hidden_layers
        self.fOut_hidden_layers = acOpt['fOut_hidden_layers'] \
            if 'fOut_hidden_layers' in acOpt.keys() else self.n_hidden_layers
        self.channels_for_nn = acOpt['channels_for_nn']

        self.channels_bypass = self.in_channels // 2
        self.channels_transformed = self.in_channels - self.channels_bypass

        self.hidden_channels = acOpt['hidden_channels']

        self.fBypassToFtNorm = self.F(in_channels=self.channels_bypass,
                                      out_channels=self.channels_transformed * 2,
                                      hidden_channels=self.channels_bypass, kernel_hidden=self.kernel_hidden,
                                      n_hidden_layers=self.fBypassToFtNorm_hidden_layers)
        self.fFtPreprocess = self.F(in_channels=self.in_channels_rrdb,
                                    out_channels=self.hidden_channels,
                                    hidden_channels=self.channels_bypass, kernel_hidden=self.kernel_hidden,
                                    n_hidden_layers=self.fFtPreprocess_hidden_layers, useZero=True)
        self.fOut = self.F(in_channels=self.hidden_channels,
                           out_channels=self.channels_transformed * 2,
                           hidden_channels=self.hidden_channels,
                           kernel_hidden=self.kernel_hidden,
                           n_hidden_layers=self.fOut_hidden_layers)

    def forward(self, input: torch.Tensor, logdet=None, reverse=False, ft=None):
        if not reverse:
            z = input
            assert z.shape[1] == self.in_channels, (z.shape[1], self.in_channels)
            z1, z2 = self.split(z)
            scale, shift = self.feature_extract(z1, ft)
            assert z1.shape[1] == self.channels_bypass, (z1.shape[1], self.channels_bypass)
            assert z2.shape[1] == self.channels_transformed, (z2.shape[1], self.channels_transformed)
            assert scale.shape[1] == shift.shape[1], (scale.shape[1], shift.shape[1])
            assert scale.shape[1] == z2.shape[1], (scale.shape[1], z1.shape[1], z2.shape[1])

            z2 = z2 + shift
            z2 = z2 * scale

            logdet = self.get_logdet(logdet, scale)
            z = thops.cat_feature(z1, z2)
            output = z
        else:
            z = input
            z1, z2 = self.split(z)
            scale, shift = self.feature_extract(z1, ft)

            z2 = z2 / scale
            z2 = z2 - shift

            logdet = -self.get_logdet(logdet, scale)
            z = thops.cat_feature(z1, z2)
            output = z
        return output, logdet

    def get_logdet(self, logdet, scale):
        logdet = thops.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        return logdet

    def feature_extract(self, z1, ft):
        hZ1 = self.fBypassToFtNorm(z1)
        shiftZ1, logScaleZ1 = thops.split_feature(hZ1, "cross")
        scaleZ1 = (torch.sigmoid(logScaleZ1 + 2.) + self.affine_eps)
        return scaleZ1, shiftZ1
        #ftPreprocessed = self.fFtPreprocess(ft)
        #z1_cond = scaleZ1 * ftPreprocessed + shiftZ1
        #z1_cond_mapped = self.fOut(z1_cond)
        #shift_z1_cond_mapped, logScale_z1_cond_mapped = thops.split_feature(z1_cond_mapped, "cross")
        #scale_z1_cond_mapped = (torch.sigmoid(logScale_z1_cond_mapped + 2.) + self.affine_eps)
        #return scale_z1_cond_mapped, shift_z1_cond_mapped

    def split(self, z):
        z1 = z[:, :self.channels_bypass]
        z2 = z[:, self.channels_bypass:]
        assert z1.shape[1] + z2.shape[1] == z.shape[1], (z1.shape[1], z2.shape[1], z.shape[1])
        return z1, z2

    def F(self, in_channels, out_channels, hidden_channels, kernel_hidden=1, n_hidden_layers=1, useZero=True):
        layers = [Conv2d(in_channels, hidden_channels), nn.ReLU(inplace=False)]

        for _ in range(n_hidden_layers):
            layers.append(Conv2d(hidden_channels, hidden_channels, kernel_size=[kernel_hidden, kernel_hidden]))
            layers.append(nn.ReLU(inplace=False))

        if useZero:
            layers.append(Conv2dZeros(hidden_channels, out_channels))
        else:
            layers.append(Conv2d(hidden_channels, out_channels))

        return nn.Sequential(*layers)


class DCl(nn.Module):
    def __init__(self, stride, in_ch):
        super().__init__()

        ly = []
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=stride, dilation=stride) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ Conv2d(in_ch, in_ch, kernel_size=1) ]
        self.body = nn.Sequential(*ly)

    def forward(self, x):
        return x + self.body(x)


class CondFeaAndCondSelf(nn.Module):
    def __init__(self, in_channels, mode, opt):
        super().__init__()
        self.need_features = True
        self.in_channels = in_channels
        self.in_channels_cond = opt_get(opt, ['flow', 'CondFeaAndCondSelf', 'in_channels_cond'])
        self.affine_eps = opt_get(opt, ['flow', 'CondFeaAndCondSelf', 'eps'],  0.0001)
        self.n_hidden_layers = opt_get(opt, ['flow', 'CondFeaAndCondSelf', 'n_hidden_layers'], 1)
        self.hidden_channels = opt_get(opt, ['flow', 'CondFeaAndCondSelf', 'hidden_channels'], 64)
        self.affine_eps = opt_get(opt, ['flow', 'CondFeaAndCondSelf', 'eps'],  0.0001)
        self.mult_reverse = opt_get(opt, ['flow', 'CondFeaAndCondSelf', 'multReverse'],  False)
        self.use_tanh = opt_get(opt, ['flow', 'CondFeaAndCondSelf', 'tanh'],  False)
        self.skip = opt_get(opt, ['flow', 'CondFeaAndCondSelf', 'skip'],  False)
        self.dilation_self_dc = opt_get(opt, ['flow', 'CondFeaAndCondSelf', 'dilation_self_DC'], 1)
        self.dilation_fea_dc = opt_get(opt, ['flow', 'CondFeaAndCondSelf', 'dilation_feature_DC'], 1)
        
        self.channels_for_nn = self.in_channels // 2
        self.channels_for_co = self.in_channels - self.channels_for_nn

        if self.channels_for_nn is None:
            self.channels_for_nn = self.in_channels // 2

        fea_layer, self_layer = mode.split('_')
        
        if fea_layer == 'FC':
            self.fFeatures = self.fc(in_channels=self.in_channels_cond,
                                     out_channels=self.in_channels * 2,
                                     hidden_channels=self.hidden_channels,
                                     n_hidden_layers=self.n_hidden_layers)
        elif fea_layer == 'C':
            self.fFeatures = self.dilated_affine(in_channels=self.in_channels_cond,
                                                 out_channels=self.in_channels * 2,
                                                 hidden_channels=self.hidden_channels,
                                                 stride=1,
                                                 n_hidden_layers=self.n_hidden_layers)
        elif fea_layer == 'DC':
            self.fFeatures = self.dilated_affine(in_channels=self.in_channels_cond,
                                                 out_channels=self.in_channels * 2,
                                                 hidden_channels=self.hidden_channels,
                                                 stride=self.dilation_fea_dc,
                                                 n_hidden_layers=self.n_hidden_layers)
        else:
            raise NotImplementedError(f'{fea_layer=}')

        if self_layer == 'FC':
            self.fAffine = self.fc(in_channels=self.channels_for_nn + self.in_channels_cond,
                                   out_channels=self.channels_for_co * 2,
                                   hidden_channels=self.hidden_channels,
                                   n_hidden_layers=self.n_hidden_layers)
        elif self_layer == 'C':
            self.fAffine = self.dilated_affine(in_channels=self.channels_for_nn + self.in_channels_cond,
                                               out_channels=self.channels_for_co * 2,
                                               hidden_channels=self.hidden_channels,
                                               stride=1,
                                               n_hidden_layers=self.n_hidden_layers)
        elif self_layer == 'DC':
            self.fAffine = self.dilated_affine(in_channels=self.channels_for_nn + self.in_channels_cond,
                                               out_channels=self.channels_for_co * 2,
                                               hidden_channels=self.hidden_channels,
                                               stride=self.dilation_self_dc,
                                               n_hidden_layers=self.n_hidden_layers)
        else:
            raise NotImplementedError(f'{fea_layer=}')

    def forward(self, input: torch.Tensor, logdet=None, reverse=False, ft=None):
        if self.skip: # quick way to deactivate conditional
            return input, logdet 

        if self.mult_reverse:
            reverse = not reverse

        if not reverse:
            z = input
            assert z.shape[1] == self.in_channels, (z.shape[1], self.in_channels)

            # Feature Conditional
            scaleFt, shiftFt = self.feature_extract(ft, self.fFeatures)
            z = z + shiftFt
            z = z * scaleFt
            logdet = logdet + self.get_logdet(scaleFt)

            # Self Conditional
            z1, z2 = self.split(z)
            scale, shift = self.feature_extract_aff(z1, ft, self.fAffine)
            self.asserts(scale, shift, z1, z2)
            z2 = z2 + shift
            z2 = z2 * scale

            logdet = logdet + self.get_logdet(scale)
            z = thops.cat_feature(z1, z2)
            output = z
        else:
            z = input

            # Self Conditional
            z1, z2 = self.split(z)
            scale, shift = self.feature_extract_aff(z1, ft, self.fAffine)
            self.asserts(scale, shift, z1, z2)
            z2 = z2 / scale
            z2 = z2 - shift
            z = thops.cat_feature(z1, z2)
            logdet = logdet - self.get_logdet(scale)

            # Feature Conditional
            scaleFt, shiftFt = self.feature_extract(ft, self.fFeatures)
            z = z / scaleFt
            z = z - shiftFt
            logdet = logdet - self.get_logdet(scaleFt)

            output = z
        return output, logdet

    def asserts(self, scale, shift, z1, z2):
        assert z1.shape[1] == self.channels_for_nn, (z1.shape[1], self.channels_for_nn)
        assert z2.shape[1] == self.channels_for_co, (z2.shape[1], self.channels_for_co)
        assert scale.shape[1] == shift.shape[1], (scale.shape[1], shift.shape[1])
        assert scale.shape[1] == z2.shape[1], (scale.shape[1], z1.shape[1], z2.shape[1])

    def get_logdet(self, scale):
        return thops.sum(torch.log(scale), dim=[1, 2, 3])

    def feature_extract(self, z, f):
        h = f(z)

        shift, scale = thops.split_feature(h, "cross")
        
        if self.use_tanh:
            scale = torch.exp(torch.tanh(scale)) + self.affine_eps
        else:
            scale = torch.sigmoid(scale + 2.) + self.affine_eps

        return scale, shift

    def feature_extract_aff(self, z1, ft, f, ):
        z = torch.cat([z1, ft], dim=1)

        h = f(z)

        shift, scale = thops.split_feature(h, "cross")

        if self.use_tanh:
            scale = torch.exp(torch.tanh(scale)) + self.affine_eps
        else:
            scale = torch.sigmoid(scale + 2.) + self.affine_eps

        return scale, shift

    def split(self, z):
        z1 = z[:, :self.channels_for_nn]
        z2 = z[:, self.channels_for_nn:]
        assert z1.shape[1] + z2.shape[1] == z.shape[1], (z1.shape[1], z2.shape[1], z.shape[1])
        return z1, z2

    def dilated_affine(self, in_channels, out_channels, hidden_channels, stride, n_hidden_layers=1):
        layers = [Conv2d(in_channels, hidden_channels, kernel_size=1), nn.ReLU(inplace=False)]

        for _ in range(n_hidden_layers):
            layers.append(DCl(stride, hidden_channels))
        layers.append(Conv2dZeros(hidden_channels, out_channels, kernel_size=1))

        return nn.Sequential(*layers)

    def fc(self, in_channels, out_channels, hidden_channels, n_hidden_layers=1):
        layers = [Conv2d(in_channels, hidden_channels, kernel_size=1), nn.ReLU(inplace=False)]
        for _ in range(n_hidden_layers):
            layers.append(Conv2d(hidden_channels, hidden_channels, kernel_size=1))
            layers.append(nn.ReLU(inplace=False))
        layers.append(Conv2dZeros(hidden_channels, out_channels, kernel_size=1))

        return nn.Sequential(*layers)


class CondFeaAffine(nn.Module):
    def __init__(self, in_channels, mode, opt):
        super().__init__()
        self.need_features = True
        self.in_channels = in_channels
        self.in_channels_cond = opt_get(opt, ['flow', 'CondFeaAndCondSelf', 'in_channels_cond'])
        self.n_hidden_layers = opt_get(opt, ['flow', 'CondFeaAffine', 'n_hidden_layers'], 1)
        self.hidden_channels = opt_get(opt, ['flow', 'CondFeaAffine', 'hidden_channels'], 64)
        self.affine_eps = opt_get(opt, ['flow', 'CondFeaAffine', 'eps'],  0.0001)
        self.mult_reverse = opt_get(opt, ['flow', 'CondFeaAffine', 'multReverse'],  False)
        self.use_tanh = opt_get(opt, ['flow', 'CondFeaAffine', 'tanh'],  False)
        self.skip = opt_get(opt, ['flow', 'CondFeaAffine', 'skip'],  False)
        self.dilation_dc = opt_get(opt, ['flow', 'CondFeaAffine', 'dilation_DC'], 1)
        
        fea_layer = mode
        if fea_layer == 'FC':
            
            self.fFeatures = self.fc(in_channels=self.in_channels_cond,
                                     out_channels=self.in_channels * 2,
                                     hidden_channels=self.hidden_channels,
                                     n_hidden_layers=self.n_hidden_layers)
        elif fea_layer == 'C':
            
            self.fFeatures = self.dilated_affine(in_channels=self.in_channels_cond,
                                                 out_channels=self.in_channels * 2,
                                                 hidden_channels=self.hidden_channels,
                                                 stride=1,
                                                 n_hidden_layers=self.n_hidden_layers)
        elif fea_layer == 'DC':
            
            self.fFeatures = self.dilated_affine(in_channels=self.in_channels_cond,
                                                 out_channels=self.in_channels * 2,
                                                 hidden_channels=self.hidden_channels,
                                                 stride=self.dilation_dc,
                                                 n_hidden_layers=self.n_hidden_layers)
        else:
            raise NotImplementedError(f'{fea_layer=}')

    def forward(self, input: torch.Tensor, logdet=None, reverse=False, ft=None):

        if self.skip: # quick way to deactivate conditional
            return input, logdet 

        if self.mult_reverse:
            reverse = not reverse

        if not reverse:
            z = input
            assert z.shape[1] == self.in_channels, (z.shape[1], self.in_channels)

            # Feature Conditional
            scaleFt, shiftFt = self.feature_extract(ft, self.fFeatures)
            z = z + shiftFt
            z = z * scaleFt
            logdet = logdet + self.get_logdet(scaleFt)

            output = z
        else:
            z = input

            # Feature Conditional
            scaleFt, shiftFt = self.feature_extract(ft, self.fFeatures)
            z = z / scaleFt
            z = z - shiftFt
            logdet = logdet - self.get_logdet(scaleFt)

            output = z
        return output, logdet

    def asserts(self, scale, shift, z1, z2):
        assert scale.shape[1] == shift.shape[1], (scale.shape[1], shift.shape[1])
        assert scale.shape[1] == z2.shape[1], (scale.shape[1], z1.shape[1], z2.shape[1])

    def get_logdet(self, scale):
        return thops.sum(torch.log(scale), dim=[1, 2, 3])

    def feature_extract(self, z, f):
        h = f(z)

        shift, scale = thops.split_feature(h, "cross")
        
        if self.use_tanh:
            scale = torch.exp(torch.tanh(scale)) + self.affine_eps
        else:
            scale = torch.sigmoid(scale + 2.) + self.affine_eps

        return scale, shift

    def dilated_affine(self, in_channels, out_channels, hidden_channels, stride, n_hidden_layers=1):
        layers = [Conv2d(in_channels, hidden_channels, kernel_size=1), nn.ReLU(inplace=False)]

        for _ in range(n_hidden_layers):
            layers.append(DCl(stride, hidden_channels))
        layers.append(Conv2dZeros(hidden_channels, out_channels, kernel_size=1))

        return nn.Sequential(*layers)

    def fc(self, in_channels, out_channels, hidden_channels, n_hidden_layers=1):
        layers = [Conv2d(in_channels, hidden_channels, kernel_size=1), nn.ReLU(inplace=False)]
        for _ in range(n_hidden_layers):
            layers.append(Conv2d(hidden_channels, hidden_channels, kernel_size=1))
            layers.append(nn.ReLU(inplace=False))
        layers.append(Conv2dZeros(hidden_channels, out_channels, kernel_size=1))

        return nn.Sequential(*layers)


class CondSelfAffine(nn.Module):
    def __init__(self, in_channels, mode, opt):
        super().__init__()
        self.need_features = True
        self.in_channels = in_channels
        self.in_channels_cond = opt_get(opt, ['flow', 'CondFeaAndCondSelf', 'in_channels_cond'])
        self.n_hidden_layers = opt_get(opt, ['flow', 'CondSelfAffine', 'n_hidden_layers'], 1)
        self.hidden_channels = opt_get(opt, ['flow', 'CondSelfAffine', 'hidden_channels'], 64)
        self.affine_eps = opt_get(opt, ['flow', 'CondSelfAffinee', 'eps'],  0.0001)
        self.mult_reverse = opt_get(opt, ['flow', 'CondSelfAffine', 'multReverse'],  False)
        self.use_tanh = opt_get(opt, ['flow', 'CondSelfAffine', 'tanh'],  False)
        self.skip = opt_get(opt, ['flow', 'CondSelfAffine', 'skip'],  False)
        self.dilation_dc = opt_get(opt, ['flow', 'CondSelfAffine', 'dilation_DC'], 1)

        self.channels_for_nn = self.in_channels // 2
        self.channels_for_co = self.in_channels - self.channels_for_nn
        
        self_layer = mode
        if self_layer == 'FC':
            
            self.fAffine = self.fc(in_channels=self.channels_for_nn + self.in_channels_cond,
                                   out_channels=self.channels_for_co * 2,
                                   hidden_channels=self.hidden_channels,
                                   n_hidden_layers=self.n_hidden_layers)
        elif self_layer == 'C':
            
            self.fAffine = self.dilated_affine(in_channels=self.channels_for_nn + self.in_channels_cond,
                                               out_channels=self.channels_for_co * 2,
                                               hidden_channels=self.hidden_channels,
                                               stride=1,
                                               n_hidden_layers=self.n_hidden_layers)
        elif self_layer == 'DC':
            
            self.fAffine = self.dilated_affine(in_channels=self.channels_for_nn + self.in_channels_cond,
                                               out_channels=self.channels_for_co * 2,
                                               hidden_channels=self.hidden_channels,
                                               stride=self.dilation_dc,
                                               n_hidden_layers=self.n_hidden_layers)
        else:
            raise NotImplementedError(f'{self_layer=}')

    def forward(self, input: torch.Tensor, logdet=None, reverse=False, ft=None):
        if self.skip: # quick way to deactivate conditional
            return input, logdet 

        if self.mult_reverse:
            reverse = not reverse

        if not reverse:
            z = input
            assert z.shape[1] == self.in_channels, (z.shape[1], self.in_channels)

            # Self Conditional
            z1, z2 = self.split(z)
            scale, shift = self.feature_extract_aff(z1, ft, self.fAffine)
            self.asserts(scale, shift, z1, z2)
            z2 = z2 + shift
            z2 = z2 * scale

            logdet = logdet + self.get_logdet(scale)
            z = thops.cat_feature(z1, z2)
            output = z
        else:
            z = input

            # Self Conditional
            z1, z2 = self.split(z)
            scale, shift = self.feature_extract_aff(z1, ft, self.fAffine)
            self.asserts(scale, shift, z1, z2)
            z2 = z2 / scale
            z2 = z2 - shift
            z = thops.cat_feature(z1, z2)
            logdet = logdet - self.get_logdet(scale)

            output = z
        return output, logdet

    def asserts(self, scale, shift, z1, z2):
        assert scale.shape[1] == shift.shape[1], (scale.shape[1], shift.shape[1])
        assert scale.shape[1] == z2.shape[1], (scale.shape[1], z1.shape[1], z2.shape[1])

    def get_logdet(self, scale):
        return thops.sum(torch.log(scale), dim=[1, 2, 3])

    def feature_extract_aff(self, z1, ft, f):
        z = torch.cat([z1, ft], dim=1)

        h = f(z)

        shift, scale = thops.split_feature(h, "cross")

        if self.use_tanh:
            scale = torch.exp(torch.tanh(scale)) + self.affine_eps
        else:
            scale = torch.sigmoid(scale + 2.) + self.affine_eps

        return scale, shift

    def split(self, z):
        z1 = z[:, :self.channels_for_nn]
        z2 = z[:, self.channels_for_nn:]
        assert z1.shape[1] + z2.shape[1] == z.shape[1], (z1.shape[1], z2.shape[1], z.shape[1])
        return z1, z2

    def dilated_affine(self, in_channels, out_channels, hidden_channels, stride, n_hidden_layers=1):
        layers = [Conv2d(in_channels, hidden_channels, kernel_size=1), nn.ReLU(inplace=False)]

        for _ in range(n_hidden_layers):
            layers.append(DCl(stride, hidden_channels))
        layers.append(Conv2dZeros(hidden_channels, out_channels, kernel_size=1))

        return nn.Sequential(*layers)

    def fc(self, in_channels, out_channels, hidden_channels, n_hidden_layers=1):
        layers = [Conv2d(in_channels, hidden_channels, kernel_size=1), nn.ReLU(inplace=False)]
        for _ in range(n_hidden_layers):
            layers.append(Conv2d(hidden_channels, hidden_channels, kernel_size=1))
            layers.append(nn.ReLU(inplace=False))
        layers.append(Conv2dZeros(hidden_channels, out_channels, kernel_size=1))

        return nn.Sequential(*layers)

