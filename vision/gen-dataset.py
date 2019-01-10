from constants import *
import cv2
from glob import glob
import pandas as pd
import numpy as np
import os

# train or test
mode = 't'
generate_patches = False

cat_file = 'catalog_{}.csv'.format(mode)
in_path = 'raw_data/'
out_path = 'patches_{}/'.format(mode)


def patch_coord(x):
    if x<= patch_size:
        return 0
    else:
        return int(np.ceil((x-patch_size)/d))


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
            # im = im[y_pad0:y_pad1, x_pad0:x_pad1] # coordinates are im[y, x]
            for y in range(0, y_size-patch_overlap, d):
                y_int = y//d
                for x in range(0, x_size-patch_overlap, d):
                    cropped_img = im[y:y+patch_size, x:x+patch_size]
                    cv2.imwrite('{}{}.{}.{}.png'.format(
                        output_folder, stripe, y_int, x//d), cropped_img)

        cat = pd.read_csv(
            '{}catalogs/SPLUS_{}_Photometry.cat'.format(input_folder, stripe),
            delimiter=' ', skipinitialspace=True, comment='#', index_col=False,
            header=None, names=cols, usecols=usecols)

        cat = cat[(cat.PROB_STAR>=prob_thres)|(cat.PROB_GAL>=prob_thres)]
        and_filters = (cat.FWHM>=fwhm_min) & (cat.FWHM<=fwhm_max) & (cat.MUMAX<=mumax_thres) & (cat.s2nDet>=s2n_thres)
        cat = cat[and_filters]
        
        cat['class'] = 'galaxy'
        cat.loc[cat.CLASS == star_int_class, 'class'] = 'star'

        cat['x0'] = cat.X - m*cat.FWHM
        cat['x1'] = cat.X + m*cat.FWHM
        cat['y0'] = y_size - cat.Y - m*cat.FWHM
        cat['y1'] = y_size - cat.Y + m*cat.FWHM

        cat.loc[cat.x0<0, 'x0'] = 0
        cat.loc[cat.x1>x_size, 'x1'] = x_size
        cat.loc[cat.y0<0, 'y0'] = 0
        cat.loc[cat.y1>y_size, 'y1'] = y_size

        cat['patch_x'] = cat.x1.apply(patch_coord)
        cat['patch_y'] = cat.y1.apply(patch_coord)

        cat['image'] = cat[['patch_y', 'patch_x']].apply(
            lambda s: '{}{}.{}.{}.png'.format(output_folder, stripe, s[0], s[1]), axis=1)

        cat['x0'] = np.floor(cat.x0 - cat.patch_x*d)
        cat['x1'] = np.ceil(cat.x1 - cat.patch_x*d)
        cat['y0'] = np.floor(cat.y0 - cat.patch_y*d)
        cat['y1'] = np.ceil(cat.y1 - cat.patch_y*d)

        int_cols = ['x0', 'x1', 'y0', 'y1']
        cat[int_cols] = cat[int_cols].astype(int)

        # zero_width = cat.x0 == cat.x1
        # cat.loc[zero_width, 'x0'] = cat.x0 - f
        # cat.loc[zero_width, 'x1'] = cat.x1 + f

        # zero_height = cat.y0 == cat.y1
        # cat.loc[zero_height, 'y0'] = cat.y0 - f
        # cat.loc[zero_height, 'y1'] = cat.y1 + f

        bounds = (cat.x0>=0) & (cat.x1<=patch_size) & (cat.y0>=0) & (cat.y1<=patch_size)
        cat = cat[bounds]

        cat_final = cat[['image', 'x0', 'y0', 'x1', 'y1', 'class']]
        cat_final.to_csv(cat_file, index=False, header=False, mode='a')


if __name__=='__main__':
    gen_data(in_path, out_path, cat_file)