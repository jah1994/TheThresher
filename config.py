#### Configuration file for The Thresher

## paths ##
#spool = '/media/jah94/Seagate_Expansion_Drive#2/Synthetic LI spool/FakeLISpool_0p6FWHM.fits' # path to data spool
#init_path = '/media/jah94/Seagate_Expansion_Drive#2/Synthetic LI spool/FakeLISpool_0p6FWHM_top50.fits' # path to image used to initialise the image model
spool = '/media/jah94/Seagate_Expansion_Drive#2/Synthetic LI spool/RealData/N7089_256x256_spool.fits' # path to data spool
init_path = '/media/jah94/Seagate_Expansion_Drive#2/Synthetic LI spool/RealData/N7089_256x256_top50_bkgsub.fits' # path to image used to initialise the image model
out_path = '/media/jah94/Seagate_Expansion_Drive#2/Synthetic LI spool/RealData/' # where to save plots and scene estimates
fname = 'scene' # scene estimates will be saved with this keyword in the filename, followed by the number of updates

## Detector parameters ##
#EMCCD_params = [60, 25, 300] # readout noise (e_EM), A/D factor (e_EM/ADU) and EM gain (e_EM / e_phot)
# 25.69693946838379, 263.044677734375, 60.963741302490234, 0.06205124408006668
EMCCD_params = [61, 25.7, 263, 0.06, 0.8] # sigma, f, gain, c, q

## SGD settings ##
proportional_clip = 5e-3 # approximate maximum fractional change in image model pixel value per update
spool_length = 4800 # number of images in spool (required as we never load the entire spool into memory)
iterations = 5 # how many iterations over the spool should be made (i.e. total number of updates = spool_length * iterations)

## Kernel (and sky) inference ##
kernel_size = 25 # single axis length of the square psf/convolution kernel object
positivity = True # non-negativity constraint on model parameters
phi = 1e-2 # hyper-parameter for controlling the strength of L1 regularisation on kernel
lr_kernel = 1e-3 # Steepest descent learning rate for kernel
lr_B = 1e-1 # Steepest descent learning rate for sky parameter
tol = 1e-9 # Minimum relative change in loss before claiming convergence
max_iters = 2500 # Maximum number of iterations for the optimisation
fisher = False # Calculate parameter uncertanties from the Fisher Information Matrix

## Housekeeping ##
sky_subtract = False # subtract median pixel value from init_path
crop_size = 0 # optional symmetrical crop to apply to images
show_convergence_plots = False # Plot (log) loss vs steps from the optimisation
show_nqrs = False
plot_freq = 100
#positive_scene = True # non-negativity constraint on image model
