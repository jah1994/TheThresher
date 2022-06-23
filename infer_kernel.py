'''
Infer the convolution kernel and sky background
'''

# imports
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# custom imports
import noise_models
import math_utils
import utils
import config

# make sure to enable GPU acceleration!
if torch.cuda.is_available() is True:
  device = 'cuda'

def inference(R, I, detector_params, loss_func, ks, positivity, phi, lr_kernel, lr_B,
                 tol, max_iters, fisher, show_convergence_plots):

    '''
    # Arguments
    * 'R' (numpy.ndarray or torch.tensor): The reference image
    * 'I' (numpy.ndarray or torch.tensor): The data/target image
    * 'detector_params' (list of floats): The readout noise (e-_EM),
       A/D conversion factor (e-_EM / ADU) and EM gain (e-_EM / e-_phot) etc.
    * 'loss_func' (function): The loss function to optimise

    # Keyword arguments
    * 'ks' (int): kernel of size ks x ks (must be be odd)
    * positivity (bool): Non-negativity constraint on model parameters
    * phi (float): hyper-parameter for controlling the strength of L1 regularisation on kernel
    * lr_kernel (float): Steepest descent learning rate for kernel
    * lr_B (float): Steepest descent learning rate for sky parameter
    * tol (float): Minimum relative change in loss before claiming convergence
    * 'max_iters' (int): Maximum number of iterations for the optimisation
    * 'fisher' (bool): Calculate parameter uncertanties from the Fisher Information Matrix
    * 'convergence_plots' (bool): Plot (log) loss vs steps from the optimisation

    # returns
    * 'kernel' (torch.Tensor): the inferred kernel
    * 'B' (float): the sky level of the data/target image
    '''

    # trim I such that target image pixels correspond to only those with valid convolution computations
    hwidth = np.int((ks - 1) / 2)
    nx, ny = I.shape
    I = I[hwidth:nx-hwidth, hwidth:nx-hwidth]

    #### Convert numpy images to tensors and move to GPU
    I, R = utils.convert_to_tensor(I), utils.convert_to_tensor(R)

    # Move to GPU if CUDA available
    time_to_move_to_GPU = time.time()
    if torch.cuda.is_available() is True:
        R = R.to(device)
        I = I.to(device)
        print("--- Time to move data onto GPU: %s ---" % (time.time() - time_to_move_to_GPU))


    # Model = kernel * R + bias (this is just a 2D convolution with an added scalar/bias)
    model = torch.nn.Sequential(
      torch.nn.Conv2d(in_channels=1,
                      out_channels=1,
                      kernel_size=ks,
                      padding = 0,
                      bias=True
                      )
    )

    # Initialise kernel and bias
    # Assumes model and data on approximately the same photomeric scale,
    # and that the model image has been background subtracted
    init_kernel_pixels = 1. / (ks**2) # ensures that the kernel sums to 1 at initialisation
    init_background = torch.median(I).item() # estimate for the 'sky' level of the target image
    model[0].weight = torch.nn.Parameter(init_kernel_pixels*torch.ones(model[0].weight.shape, requires_grad=True))
    model[0].bias = torch.nn.Parameter(init_background*torch.ones(model[0].bias.shape, requires_grad=True))

    # Move model to GPU if available
    if torch.cuda.is_available() is True:
      model = model.to(device)

    ## Setup the optimsation
    # Keep track of the loss
    losses = []

    # Initialise Adam optimiser
    optimizer_Adam = torch.optim.Adam([
                  {'params': model[0].weight, 'lr': lr_kernel},
                  {'params': model[0].bias, 'lr': lr_B}
              ])

    # Time the optimisation
    start_time_infer = time.time()

    torch.set_printoptions(precision=10)
    print('Check dtype of data and weights:')
    print(R.dtype, I.dtype, model[0].weight.dtype, model[0].bias.dtype)
    print('Check size of data and weights:')
    print(R.size(), I.size(), model[0].weight.size(), model[0].bias.size())


    # Optimise!
    print('Starting optimisation...')
    for t in range(max_iters):

        # compute forward model
        y_pred = model(R)

        # compute the loss
        loss = loss_func(y_pred, I, detector_params, phi, model[0].weight)

        # print iters vs. loss at set interval - useful for reviewing learning rate choices
        if t % 50 == 0:
            print('Iteration:%d, loss=%f, P=%f, B=%f' % (t, loss.item(), torch.sum(model[0].weight).item(), model[0].bias.item()))

        # clear gradients, compute gradients, take a single
        # steepest descent step
        optimizer_Adam.zero_grad()
        loss.backward()
        optimizer_Adam.step()

        # non-negativity
        if positivity == True:
          with torch.no_grad():
            model[0].weight.clamp_(min=0)
            model[0].bias.clamp_(min=0)

        # track losses
        losses.append(loss.detach())


        # Convergence reached if less than specified tol and more than 10
        # steps taken (guard against early stopping)
        if t>10 and abs((losses[-1] - losses[-2])/losses[-2]) < tol:
            print('Converged!')
            print('Total steps taken:', t)
            print("--- Finished kernel and background fit in %s seconds ---" % (time.time() - start_time_infer))
            break

        # if likelihood evaluation fails, break
        elif torch.isnan(losses[-1]) == True:
            break

        # if we reach max_iters, break
        elif t == max_iters - 1:
            print('Failed to converge within the designated maximum number of steps!')
            break


    kernel, B = model[0].weight, model[0].bias

    ### Optional - Estimate parameter uncertanties via Fisher Information
    if fisher == True:

        y_pred = model(R)
        if config.EMCCD == True:
            loss = noise_models.emccd_nll(y_pred, I, EMCCD_params, phi, model[0].weight)
        elif config.CCD == True:
            loss = noise_models.ccd_nll(y_pred, I, CCD_params, phi, model[0].weight)
        else:
            print('No noise model specified!')
        logloss_grads = torch.autograd.grad(loss, model.parameters(), create_graph=True, retain_graph=True)
        print('Building full Hessian...')
        full_hessian_time = time.time()
        H = math_utils.compute_full_hessian(logloss_grads)
        print('---Finished constructing full hessian in %s seconds---' % (time.time() - full_hessian_time))
        cov_matrix_diags = math_utils.get_stderror(H)
        psf_err, B_err = torch.sqrt(torch.sum(cov_matrix_diags[0:-(2*d+1)])), torch.sqrt(cov_matrix_diags[-(2*d+1)])
        print('Photometric scale factor:', torch.sum(kernel).item(), '+/-', psf_err.item())
        print('Sky:', B.item(), '+/-', B_err.item())

    else:
        print('Photometric scale factor:', torch.sum(kernel).item())
        print('Sky:', B.item())


    if show_convergence_plots == True:
        fig = plt.figure(figsize=(10, 7))
        plt.plot(np.log10(losses))
        plt.xlabel('Iterations')
        plt.ylabel('(log10) Loss')
        plt.grid()
        plt.savefig(os.path.join(config.out_path, 'plots', 'Loss.png'), bbox_inches='tight');
        fig.clear()
        plt.close(fig)
        plt.clf()

    if config.show_nqrs == True:
        math_utils.compute_nqr(I, y_pred)

    return kernel.detach().cpu(), B.detach().cpu()
