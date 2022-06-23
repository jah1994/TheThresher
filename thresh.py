'''
The Thresher is a tool for fitting image models to spools of high
frame-rate imaging data.
'''

## imports
# standard imports
import torch
import numpy as np
from astropy.io import fits
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

# custom imports
import infer_kernel
import noise_models
import utils
import math_utils
import config

# check torch version
print('PyTorch version:', torch.__version__)

# make sure to enable GPU acceleration!
if torch.cuda.is_available() is True:
  device = 'cuda'
else:
  print('GPU not found, defaulting to CPU (not thoroughly tested!)...')

# create directory in which to save progress plots
Path(os.path.join(config.out_path, 'plots')).mkdir(parents=True, exist_ok=True)
# create directory in which to save image model estimates
Path(os.path.join(config.out_path, 'scene_estimates')).mkdir(parents=True, exist_ok=True)


# Initialise the image model
scene = fits.getdata(config.init_path) # load fits
scene = utils.crop(scene, config.crop_size) # crop
if config.sky_subtract is True: # OPTIONAL sky subtraction
    print('Subtracting median pixel value from the scene initialisation')
    scene -= np.median(scene)
else:
    print('No sky subtraction...')

scene[scene < 0.] = 0. # positivity
s0 = np.copy(scene) # store a copy of the initialisation
scene = utils.convert_to_tensor(scene) # convert scene to tensor
alpha0 = config.proportional_clip * torch.clone(scene) # initalise step-size
alpha0[alpha0 == 0] = torch.min(alpha0[alpha0 != 0]) # avoid 'zero' updates

scene.requires_grad = True # we want gradients of our model parameters

# adopt an EMCCD noise model
if config.EMCCD == True:
    detector_params = config.EMCCD_params
    loss_func = noise_models.emccd_nll

# adopt a CCD noise model
elif config.CCD == True:
    detector_params = config.CCD_params
    loss_func = noise_models.ccd_nll

else:
    print('No noise model specified!')

# initialise the update counter
c = 0

# make multiple passes over the data
for p in range(config.iterations):
    # online optimisation: load a single 'n' image at a time
    for i in range(config.spool_length):

        print('Image %d/%d' % (i+1, config.spool_length))
        print('Number of updates:', c)

        y = fits.getdata(config.spool)[i] # load ith image
        y = utils.crop(y, config.crop_size) # apply crop
        y = utils.align(s0, y) # integer align

        # infer psf/kernel and the differential background
        try:

            psf, sky  = infer_kernel.inference(scene,
                                                y,
                                                detector_params,
                                                loss_func,
                                                ks = config.kernel_size,
                                                positivity = config.positivity,
                                                phi = config.phi,
                                                lr_kernel = config.lr_kernel,
                                                lr_B = config.lr_B,
                                                tol = config.tol,
                                                max_iters = config.max_iters,
                                                fisher = config.fisher,
                                                show_convergence_plots = config.show_convergence_plots)

        except RuntimeError:
            continue

        if torch.isnan(psf).any() == False:

            print('PSF inference successful... updating image model')

            # convert y to tensor
            y = utils.convert_to_tensor(y)

            # compute forward model
            prediction = torch.nn.functional.conv2d(scene, psf, bias=sky,
                               padding=int(((config.kernel_size - 1)/2)))

            # compute the loss (negative log-likelihood)
            loss = loss_func(prediction, y, detector_params)

            # compute gradients
            dl_ds = torch.autograd.grad(loss, scene)[0]

            if torch.isnan(dl_ds).any() == False:

                # initialise 1st and 2nd moment vectors
                if c == 0:
                    m = torch.zeros(scene.size())
                    v = torch.zeros(scene.size())

                update, m, v = math_utils.Adam(c, dl_ds, m, v, alpha=alpha0)

                scene = scene - update

                # update counter
                c += 1

            else:
                continue

            # enforce positivity (+ sky subtract)
            with torch.no_grad():
                scene.clamp_(min=0)

            if c % config.plot_freq == 0 or c == 1:

                # track progress
                fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 10))

                ax[0].imshow(psf[0][0], origin='lower')
                ax[0].text(8, 25, 'Convolution Kernel')
                cbar0 = plt.colorbar(ax[1].imshow(psf[0][0]), ax=ax[0], fraction=0.046, pad=0.04)

                ax[1].imshow(-update[0][0], origin='lower')
                ax[1].text(32, -3, 'Update')
                cbar1 = plt.colorbar(ax[1].imshow(-update[0][0]), ax=ax[1], fraction=0.046, pad=0.04)

                ax[2].imshow(scene[0][0].detach().numpy() - s0, norm=SymLogNorm(0.1))
                ax[2].text(21, -3, 'Difference from initialisation (log10)')
                cbar2 = plt.colorbar(ax[2].imshow(scene[0][0].detach().numpy() - s0, norm=SymLogNorm(0.1)),
                        ax=ax[2], fraction=0.046, pad=0.04)

                plt.savefig(os.path.join(config.out_path, 'plots', 'Progress_plot_%d.png' % c),
                            bbox_inches='tight');

                fig.clear()
                plt.close(fig)
                plt.clf()

                fname = os.path.join(config.out_path, 'scene_estimates', config.fname + '_' + str(c) + '.fits')
                utils.save_numpy_as_fits(scene[0][0].detach().numpy(), fname)
