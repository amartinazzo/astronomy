from constants import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

input_folder = 'raw_data/'
stripe = 'STRIPE82-0003'

cat = pd.read_csv('{}catalogs/SPLUS_{}_Photometry.cat'.format(input_folder, stripe),
    delimiter=' ', skipinitialspace=True, comment='#', index_col=False, header=None,
    names=cols, usecols=usecols)

# print(cat.sort_values(by='s2nDet', ascending=False))

fig, ax = plt.subplots()
cat[['s2nDet', 'FWHM', 'MUMAX']].hist(ax=ax,bins=np.linspace(0,50,num=100))
plt.show()