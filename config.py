#### Configuration file for The Thresher

## paths ##
#spool = '/media/jah94/Seagate_Expansion_Drive#2/Synthetic LI spool/FakeLISpool_0p6FWHM.fits'
#init_path = '/media/jah94/Seagate_Expansion_Drive#2/Synthetic LI spool/FakeLISpool_0p6FWHM_top50.fits'
#out_path = '/media/jah94/Seagate_Expansion_Drive#2/Synthetic LI spool/nocosmic/'
spool = '/media/jah94/Seagate_Expansion_Drive#2/Synthetic LI spool/RealData/N7089_256x256_spool_nocosmics.fits' # path to data spool
init_path = '/media/jah94/Seagate_Expansion_Drive#2/Synthetic LI spool/RealData/N7089_256x256_top50_bkgsub.fits' # path to image used to initialise the image model
out_path = '/media/jah94/Seagate_Expansion_Drive#2/Synthetic LI spool/RealData/nocosmic/' # where to save plots and scene estimates
#spool = '/media/jah94/Seagate_Expansion_Drive#2/2019to2020_NGC4395_PowerSpecImages/NGC4395.fits'
#init_path = '/media/jah94/Seagate_Expansion_Drive#2/2019to2020_NGC4395_PowerSpecImages/NGC4395_ref.fits'
#out_path = '/media/jah94/Seagate_Expansion_Drive#2/2019to2020_NGC4395_PowerSpecImages/'

fname = 'scene' # scene estimates will be saved with this keyword in the filename, followed by the number of updates

## Detector parameters ##
EMCCD = True
EMCCD_params = [60.39, 25.47, 259.71, 0.06, 0.8] # sigma, f, (EM) gain, c, q (check noise_models for units)
#EMCCD_params = [60, 25, 300, 0, 1] # sigma, f, (EM) gain, c, q (check noise_models for units)

CCD = False
CCD_params = [10.5/7.7, 7.7] # sigma, gain

## SGD settings ##
proportional_clip = 5e-3 # approximate maximum fractional change in image model pixel value per update
spool_length = 4800 # number of images in spool (required as we never load the entire spool into memory)
iterations = 1 # how many iterations over the spool should be made (i.e. total number of updates = spool_length * iterations)

## Kernel (and sky) inference ##
kernel_size = 25 # single axis length of the square psf/convolution kernel object
positivity = True # non-negativity constraint on model parameters
phi = 0.03 # hyper-parameter for controlling the strength of L1 regularisation on kernel
lr_kernel = 1e-3 # Steepest descent learning rate for kernel
lr_B = 0.1 # Steepest descent learning rate for sky parameter
tol = 1e-9 # Minimum relative change in loss before claiming convergence
max_iters = 2500 # Maximum number of iterations for the optimisation
fisher = False # Calculate parameter uncertanties from the Fisher Information Matrix

## Housekeeping ##
sky_subtract = False # subtract median pixel value from init_path
crop_size = 0 # optional symmetrical crop to apply to images
show_convergence_plots = False # Plot (log) loss vs steps from the optimisation
show_nqrs = False
plot_freq = 100
