from constants import *
import cv2
from glob import glob
import pandas as pd
import numpy as np
import os

# train or test
mode = 'train'
generate_patches = False

cat_file = 'catalog_{}.csv'.format(mode)
in_path = 'raw_data/'
out_path = 'patches_{}/'.format(mode)

def patch_coord(x):
    if x<= patch_size:
        return 0
    else:
        return int(np.ceil((x-patch_size)/patch_step))


def gen_data(input_folder, output_folder, csv_file):
    if os.path.exists(cat_file):
        print('error: catalog file already exists')
        return 1

    if generate_patches:
        if os.path.isdir(output_folder):
            print('error: output folder already exists')
            return 1
        os.mkdir(output_folder)

    files = glob('{}{}_images/*'.format(input_folder, mode), recursive=True)
    n_files = len(files)

    for ix, filename in enumerate(files):
        print('{}/{} processing {}...'.format(ix+1, n_files, filename))
        stripe = filename.split('/')[-1].split('.')[0]

        if generate_patches:
            im = cv2.imread(filename)
            im = im[pad_y0:pad_y1, pad_x0:pad_x1] # coordinates are im[y, x]
            for y in range(0, pad_y1-pad_y0-patch_overlap, patch_step):
                y_int = y//patch_step
                for x in range(0, pad_x1-pad_x0-patch_overlap, patch_step):
                    cropped_img = im[y:y+patch_size, x:x+patch_size]
                    cv2.imwrite('{}{}.{}.{}.png'.format(
                        output_folder, stripe, y_int, x//patch_step), cropped_img)

        cat = pd.read_csv(
            '{}catalogs/SPLUS_{}_Photometry.cat'.format(input_folder, stripe),
            delimiter=' ', skipinitialspace=True, comment='#', index_col=False,
            header=None, names=cols, usecols=usecols)

        cat = cat[(cat.PROB_STAR>=prob_min)|(cat.PROB_GAL>=prob_min)]
        and_filters = (cat.MUMAX>=mumax_min) & (cat.MUMAX<=mumax_max) & (cat.PhotoFlag==photoflag)
        cat = cat[and_filters]
        
        cat['class'] = 'galaxy'
        cat.loc[cat.CLASS == star_int_class, 'class'] = 'star'

        cat['X'] = cat.X - pad_x0
        cat['Y'] = img_size-pad_y0 - cat.Y

        cat['patch_x'] = cat.X.apply(patch_coord)
        cat['patch_y'] = cat.Y.apply(patch_coord)

        cat.loc[cat.FWHM < f, 'FWHM'] = f
        cat['x0'] = cat.X - m*cat.FWHM - cat.patch_x*patch_step
        cat['x1'] = cat.X + m*cat.FWHM - cat.patch_x*patch_step
        cat['y0'] = cat.Y - m*cat.FWHM - cat.patch_y*patch_step
        cat['y1'] = cat.Y + m*cat.FWHM - cat.patch_y*patch_step

        # cat.loc[cat.x0<0, 'x0'] = 0
        # cat.loc[cat.x1>img_size, 'x1'] = img_size
        # cat.loc[cat.y0<0, 'y0'] = 0
        # cat.loc[cat.y1>img_size, 'y1'] = img_size

        cat['image'] = cat[['patch_y', 'patch_x']].apply(
            lambda s: '{}{}.{}.{}.png'.format(output_folder, stripe, s[0], s[1]), axis=1)

        int_cols = ['x0', 'x1', 'y0', 'y1']
        cat[int_cols] = cat[int_cols].astype(int)

        bounds = (cat.x0>=0) & (cat.x1.between(cat.x0+1, patch_size)) & (cat.y0>=0) & (cat.y1.between(cat.y0+1,patch_size))
        cat = cat[bounds]
        cat = cat[['image', 'x0', 'y0', 'x1', 'y1', 'class']]

        cat.to_csv(cat_file, index=False, header=False, mode='a')

    d = pd.read_csv(cat_file, header=None)
    msk = d[0].str.contains(".13.png")
    d = d[~msk]
    d.to_csv(cat_file, header=None, index=False)



if __name__=='__main__':
    gen_data(in_path, out_path, cat_file)