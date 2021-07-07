# imports
import numpy as np
import torch
from astropy.io import fits
from skimage.feature import register_translation
from scipy.ndimage import shift

# Convert np.ndarrays to torch.Tensors whith dims: NHWC
def convert_to_tensor(image):
  if type(image) is np.ndarray:
      image = image.astype(np.float32)
      image = torch.tensor(image[None, None, :, :])
  else:
      pass
  return image

def crop(image, crop_size):
    # symmetric crop around image border
    return image[crop_size:image.shape[0]-crop_size, crop_size:image.shape[1]-crop_size]

def save_numpy_as_fits(numpy_array, filename):
    hdu = fits.PrimaryHDU(numpy_array)
    hdul = fits.HDUList([hdu])
    hdul.writeto(filename, overwrite=True)

# integer alignment
def align(template, data):
    shifts, error, diffphase = register_translation(template, data, 100)
    xs, ys = shifts[1], shifts[0]
    aligned_data = shift(data, (int(ys), int(xs)), order=0, cval=0)
    return aligned_data
