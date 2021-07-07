#### Configuration file for The Thresher

spool = '/media/jah94/Seagate_Expansion_Drive#2/Synthetic LI spool/10_source_spool.fits' # path to data spool
spool_length = 10 # number of images in spool
init_path = '/media/jah94/Seagate_Expansion_Drive#2/Synthetic LI spool/top50_15sources.fits' # path to image used to initialise the image model
crop_size = 0 # optional symmetrical crop to apply to images
#positive_scene = True # non-negativity constraint on image model

out_path = '/media/jah94/Seagate_Expansion_Drive#2/Synthetic LI spool/'
proportional_clip = 1e-3 # approximate max fractional change in model pixel value per update
iterations = 3
plot_freq = 1
fname = 'scene' # scene estimates will be saved with this keyword in the filename

####################################
## Kernel (and sky) inference ##
EMCCD_params = [60, 25.8, 300] # readout noise (e_EM), A/D factor (e_EM/ADU) and EM gain (e_EM / e_phot)
kernel_size = 25 # single axis length of the square psf/convolution kernel object
positivity = True # non-negativity constraint on model parameters
phi = 1e-3 # hyper-parameter for controlling the strength of L1 regularisation on kernel
lr_kernel = 1e-3 # Steepest descent learning rate for kernel
lr_B = 1e-1 # Steepest descent learning rate for sky parameter
tol = 1e-9 # Minimum relative change in loss before claiming convergence
max_iters = 1000 # Maximum number of iterations for the optimisation
fisher = False # Calculate parameter uncertanties from the Fisher Information Matrix
show_convergence_plots = True # Plot (log) loss vs steps from the optimisation
####################################
