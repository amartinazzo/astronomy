import numpy as np


def hms2dec(h, m, s):
  return 15*(h + m/60 + s/3600)


def dms2dec(d, m, s):
  if d<0:
    return (d - m/60 - s/3600)
  else:
    return (d + m/60 + s/3600)


def angular_dist(ra1, dec1, ra2, dec2):
  ra1 = np.radians(ra1)
  dec1 = np.radians(dec1)
  ra2 = np.radians(ra2)
  dec2 = np.radians(dec2)
  a = np.sin(np.abs(dec1-dec2)/2)**2
  b = np.cos(dec1) * np.cos(dec2) * np.sin(np.abs(ra1 - ra2)/2)**2
  return np.degrees(2*np.arcsin(np.sqrt(a + b)))


##TODO download super catalogue
def import_super():
  cat = np.loadtxt('super.csv', delimiter=',', skiprows=1, usecols=[0, 1])
  final_cat = []
  i = 1
  for obj in cat:
    t = (i, obj[0], obj[1])
    final_cat.append(t)
    i+=1
  return final_cat


def find_closest(cat, ra, dec):
  dmin = 1000
  closest = 0
  for obj in cat:
    d = angular_dist(obj[1], obj[2], ra, dec)
    if d<dmin:
      dmin = d
      closest = obj[0]
  return (closest, dmin)

def naive_crossmatch(cat1, cat2, max_dist):
  matches = []
  no_matches = []
  for obj in cat1:
    (closest_id, d) = find_closest(cat2, obj[1], obj[2])
    if d < max_dist:
      matches.append((obj[0], closest_id, d))
    else:
      no_matches.append(obj[0])
  return (matches, no_matches)
