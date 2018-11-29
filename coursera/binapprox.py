import numpy as np

# this is an implementation of the binapprox algorithm, found in:
# http://www.stat.cmu.edu/~ryantibs/papers/median.pdf

# receives a set of values and a number of bins B
# returns a tuple: (mean of values, std of values, count of values smaller than minval, array of counts)
def median_bins(values, B):
  mean = np.mean(values)
  std = np.std(values)
  minval = mean - std
  maxval = mean + std
  bin_width = 2*std/B
  bin_counts = np.zeros(B)
  minrange = minval
  for i in range(B):
    inrange = [v for v in values if minrange <= v < minrange + bin_width]
    bin_counts[i] = len(inrange)
    minrange = minrange + bin_width
  ignored_bin = [v for v in values if v < minval]
  return (mean, std, len(ignored_bin), bin_counts)

# receives a set of values and a number of bins B
# returns estimated median
def median_approx(values, B):
  (mean, std, ignored_count, bin_counts) = median_bins(values, B)
  bin_width = 2*std/B
  minval = mean - std
  N = len(values)
  threshold = (N+1)/2
  total = ignored_count
  for i in range(B):
    total += bin_counts[i]
    midpoint = minval + i*bin_width + bin_width/2
    if total >= threshold:
      break
  return midpoint


  # TODO: expand functions to receive 2d arrays (FITS images)


if __name__ == '__main__':
  print(median_bins([1, 1, 3, 2, 2, 6], 3))
  print(median_approx([1, 1, 3, 2, 2, 6], 3))

  print(median_bins([1, 5, 7, 7, 3, 6, 1, 1], 4))
  print(median_approx([1, 5, 7, 7, 3, 6, 1, 1], 4))
