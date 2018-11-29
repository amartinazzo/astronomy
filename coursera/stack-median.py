from astropy.io import fits
import matplotlib.pyplot as plot
import numpy as np
import time

# receives a function, a size (int), and a number of trials (int)
# returns time elapsed to run function a given number of trials with a random array of given size
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

# receives a csv file
# returns its mean and median
def calc_stats(file):
  data = np.loadtxt(file, delimiter=',')
  return (np.round(np.mean(data), 1), np.round(np.median(data), 1))

# receives a list of csv files
# returns an array with mean values
def mean_datasets(datasets):
  data = np.loadtxt(datasets[0], delimiter=',')
  for i in range(1, len(datasets)):
    data = data + np.loadtxt(datasets[i], delimiter=',')
  return np.round(data / len(datasets), 1)

# receives a fits image
# returns coordinates of the brightest pixel
def brightest_pixel(file):
  hdulist = fits.open(file)
  data = hdulist[0].data
  x_max = np.argmax(np.max(data, axis=1))
  y_max = np.argmax(np.max(data, axis=0))
  return (x_max, y_max)

# receives a set of fits images
# returns a matrix of mean brightness values
def mean_fits(image_set):
  hdulist = fits.open(image_set[0])
  data = hdulist[0].data
  for i in range(1, len(image_set)):
    hdulist = fits.open(image_set[i])
    data = data + hdulist[0].data
  return data/len(image_set)

# receives a set of fits images
# returns a tuple: (median of stacked images, time elapsed, memory used)
def median_fits(image_set):
  start = time.perf_counter()
  image_count = len(image_set)
  hdulist = fits.open(image_set[0])
  input_matrix = np.matrix(hdulist[0].data)
  (lines, columns) = input_matrix.shape
  stack_matrix = np.zeros((image_count, columns, lines))
  stack_matrix[0] = input_matrix
  for z in range(1, image_count):
    hdulist = fits.open(image_set[z])
    input_matrix = np.matrix(hdulist[0].data)
    stack_matrix[z] = input_matrix
  median = np.median(stack_matrix, axis=0)
  duration = time.perf_counter() - start
  memory = sys.getsizeof(stack_matrix)/1024 #in kB
  return (median, duration, memory)

if __name__ = '__main__':
  start = time.perf_counter()
  # do slow stuff
  end = time.perf_counter() - start
  