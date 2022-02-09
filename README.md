# TheThresher

This is a tool for online blind deconvolution of high frame-rate astronomical imaging data.

It attempts to fit a model to large stacks of imaging data, in which the image-to-image PSF is varying, via a stochastic optimisation.

It is a continuation of DWH and DFM's work on a project of the same name, https://github.com/davidwhogg/TheThresher.

## How to run this software

For now, you can just download TheThresher directory.

After you've setup the configuration file (see below), just run the following from the command line:

  python thresh.py

Estimates of the model will be continually saved to some specified directory as the optimisation proceeds.

## The configuration file

The file that needs editing by the user is config.py. The variables are explained below, with some example entries:

  ## Paths ##
  spool = 'MyLISpool.fits' # path to the image spool (should be a .fits file)
  init_path = 'ModelInitialisation.fits' # the image used to initialise the model (should be a .fits file)
  out_path = '/DirectoryOfResults/' # where the continually updated output from the stochastic optimsiation is stored

  fname = 'scene' # scene estimates will be saved with this keyword in the filename, followed by the number of updates

  ## Detector parameters ##
  # The Thresher supports noise models for both EMCCD and CCD images. The user must specify
  # which to use (e.g. EMCCD = True), and then supply some appropriate noise model parameters. For example, below,
  # the EMCCD noise model will be adopted. The the CCD flag is set to False to make sure its ignored.
  EMCCD = True
  EMCCD_params = [60, 25, 300, 0, 1] # readout noise, f, (EM) gain, c, q (check noise_models for units)

  CCD = False
  CCD_params = [10.5/7.7, 7.7] # readout noise (ADU), gain (e-/ADU)

  ## SGD settings ##
  # The proportional_clip may require at bit of tinkering to get good results, but values between 1e-3 to 1e-1
  # usually work OK on most problems we've encountered.
  proportional_clip = 5e-3 # approximate maximum fractional change in image model pixel value per update
  spool_length = 3000 # number of images in spool (required as we never load the entire spool into memory)
  iterations = 1 # how many iterations over the spool should be made (i.e. total number of updates = spool_length * iterations)

  ## Kernel and sky fit ##
  # Again, a bit of tinkering will be necessary here to get the best performance. Good values for lr_kernel are
  # usually somewhere in the range of 1e-3 - 1e-2, and suggested values for lr_B are within 1e-1 - 1e1.
  # phi also may require careful tuning, but values between 1e-3 - 1e-1 usually work OK.
  # And if the initialisation for the model has been background subtracted, it's strongly
  # recommended to keep positivity = True for numerical stability
  kernel_size = 25 # single axis length of the square psf/convolution kernel object
  positivity = True # non-negativity constraint on kernel and sky parameters
  phi = 0.07 # hyper-parameter for controlling the strength of L1 regularisation on kernel
  lr_kernel = 1e-3 # Steepest descen (Adam) learning rate for kernel
  lr_B = 0.1 # Steepest descent (Adam) learning rate for sky parameter
  tol = 1e-9 # Minimum relative change in loss before claiming convergence
  max_iters = 2500 # Maximum number of iterations for the optimisation
  fisher = False # Calculate parameter uncertanties from the Fisher Information Matrix (advised to keep this False)

  ## Housekeeping ##
  sky_subtract = True # subtract median pixel value from init_path
  crop_size = 0 # optional symmetrical crop to apply to images
  show_convergence_plots = False # Plot (log) loss vs steps from the optimisation
  show_nqrs = False # plot up the normalised quantile residuals from the kernel and sky fit
  plot_freq = 100 # the interval between which model estimates and plots are saved into out_path


## Get in touch

Optimisation problems are hard! For any advice in hyperparameter tuning, or any general queries, please do feel free to get in touch at: jah36@st-andrews.ac.uk
