from astropy.io import fits
import matplotlib.pyplot as plot
import numpy as np

# load a csv file and return its mean and median
def calc_stats(file):
  data = np.loadtxt(file, delimiter=',')
  return (np.round(np.mean(data), 1), np.round(np.median(data), 1))

# load a list of csv files and return an array with mean values
def mean_datasets(datasets):
  data = np.loadtxt(datasets[0], delimiter=',')
  for i in range(1, len(datasets)):
    data = data + np.loadtxt(datasets[i], delimiter=',')
  return np.round(data / len(datasets), 1)

# load a fits image and return coordinates of the brightest pixel
def brightest_pixel(file):
  hdulist = fits.open(file)
  data = hdulist[0].data
  x_max = np.argmax(np.max(data, axis=1))
  y_max = np.argmax(np.max(data, axis=0))
  return (x_max, y_max)

# load a list of fits images, stack them, and return an array with mean brightness values
def mean_fits(image_set):
  hdulist = fits.open(image_set[0])
  data = hdulist[0].data
  for i in range(1, len(image_set)):
    hdulist = fits.open(image_set[i])
    data = data + hdulist[0].data
  return data/len(image_set)

# run and test functions
if __name__ == '__main__':
  
  # fits_data  = mean_fits(['image0.fits', 'image1.fits', 'image2.fits'])