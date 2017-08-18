from astropy.io import fits
import matplotlib.pyplot as plot
import numpy as np
import time

# return the time elapsed to run a function with a random array of given size and number of trials
def time_count(function, size, ntrials):
  data = np.random.rand(size)
  total_time = 0
  for i in range(0, ntrials):
    start = time.perf_counter()
    res = func(data)
    end = time.perf_counter() - start
    total_time += end
    data = np.random.rand(size)
  return total_time/ntrials

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

if __name__ = '__main__':
  start = time.perf_counter()
  # do slow stuff
  end = time.perf_counter() - start
  