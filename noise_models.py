import torch
import numpy as np
import math_utils

# EMCCD noise model - negative log-likelihood
def emccd_nll(model, targ, EMCCD_params, coeffs, phi = 0, w = None):

    ## units
    # g (e-_EM), electrons generated after the EM amplification
    # n (e-_phot), electron generated before the EM amplification
    # sigma (e-_EM), the standard deviation of the readnoise
    # gain (e-_EM / e-_phot), EM amplification gain
    # f (e-_EM / ADU)
    # counts (ADU) i.e. image pixel values

    # EMCCD detector parameters
    sigma, f, gain = EMCCD_params

    # collapse everything to 1D
    model, targ = model.flatten(), targ.flatten()

    # for convenience, sort targ, and realign model so that pixel pairs correspond
    targ, indices = targ.sort()
    model = model[indices]

    # define 'pi' as a tensor
    pi = torch.from_numpy(np.array(np.pi))

    # convert ADU counts to appropriate dimensions
    g = f*targ # (e-_EM / ADU) * ADU = e-_EM
    n = (f/gain)*model # (e-_EM / ADU) * (e-_phot / e_EM) * ADU = e-_phot

    ##### gaussian read noise #####
    pdf_readout = torch.exp(-n) * (1./torch.sqrt(2*pi*sigma**2)) * torch.exp(-0.5*(g/sigma)**2)

    ##### EM gain noise #####
    # only defined for g>0
    g_pos = g[g>0]
    n_pos = n[g>0]

    # require n_pos > 0
    n_pos = torch.clamp(n_pos, min=1e-34)

    # evaluate modified bessel function of first order
    x = 2*torch.sqrt((n_pos*g_pos)/gain)
    bessel = math_utils.i1_vectorised(x, coeffs)

    # EM pdf
    pdf_EM = torch.exp(-n_pos - (g_pos/gain)) * torch.sqrt(n_pos/(g_pos*gain)) * bessel
    pdf_EM = torch.clamp(pdf_EM, min = 1e-34)

    # add EM pdf to readout pdf for g>0 pixels
    pdf_pos = pdf_readout[g > 0] + pdf_EM
    pdf_neg = pdf_readout[g <= 0]

    # plug everything back together and compute the log-likelihood
    pdf = f*torch.cat((pdf_neg, pdf_pos)) # convert to 1/ADU = (e-_EM/ADU) * (1/e-_EM)
    ll = torch.sum(torch.log(pdf))

    # L1 norm on kernel
    if phi != 0.:
        vector = w[0][0].flatten()
        N_dat = targ.size()[0]
        prior = -phi * N_dat * torch.sum(torch.abs(vector))
        ll += prior

    return -ll
