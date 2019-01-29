from constants import *
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

input_folder = 'raw_data/catalogs/'

list_ = []
files = glob.glob(input_folder + "*.cat")
n_files = len(files)

for i, file_ in enumerate(files):
    print("{} reading {}".format(i/n_files, file_))
    df = pd.read_csv(file_,
        delimiter=' ', skipinitialspace=True, comment='#', index_col=False,
        header=None, names=cols, usecols=usecols)
    df = df[['s2nDet', 'FWHM', 'MUMAX']]
    list_.append(df)

frame = pd.concat(list_, axis=0, ignore_index=True)
print(frame.shape)

print(frame[frame.FWHM>=200].shape)
# print(frame.sort_values(by='FWHM', ascending=False).head(400))

fig, ax = plt.subplots(ncols=1, nrows=2)
for i, a in enumerate(ax):
    ax[i].ticklabel_format(style='sci', scilimits=(-1,2))
    ax[i].tick_params(labelsize=8)

frame[['s2nDet', 'FWHM']].hist(ax=ax,bins=np.linspace(0,50,num=100), color='gray')
plt.savefig('prop_distribuitions_2.png')
plt.show()