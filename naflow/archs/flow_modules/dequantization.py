import torch
import numpy as np
import archs.flow_modules.thops as thops

def uniform_dequantization(z, logdet, quant=256):     
    z = z + ((torch.rand(z.shape, device=z.device) - 0.5) / quant)
    logdet = logdet + float(-np.log(quant) * thops.pixels(z))

    return z, logdet
