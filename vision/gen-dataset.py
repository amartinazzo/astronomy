import cv2
from glob import glob
import pandas as pd
import numpy as np
import os


cat_file = 'catalog.csv'
in_path = 'raw_data/'
out_path = 'train_patches/'

patch_size = 1000
prob_thres = .9
m = 2               # multiply by FWHM to generate bounding boxes


def gen_data(input_folder, output_folder, csv_file):
    if os.path.isdir(output_folder):
        print('error while generating data: output folder already exists')
        return 1

    os.mkdir(output_folder)

    for filename in glob(input_folder+'*.png'):
        stripe = filename.split('/')[-1].split('.')[0]

        im = cv2.imread(filename)
        imshape = im.shape

        for y in range(0, imshape[0], patch_size):
            y_int = y//patch_size
            for x in range(0, imshape[1], patch_size):
                cropped_img = im[y:y+patch_size, x:x+patch_size]
                cv2.imwrite('{}{}.{}.{}.png'.format(
                    output_folder, stripe, y_int, x//patch_size), cropped_img)

        cat = pd.read_csv('{}SPLUS_{}_Photometry.cat'.format(input_folder, stripe),
            delimiter=' ', skipinitialspace=True, comment='#', index_col=False)

        cat = cat[(cat.PROB_STAR>prob_thres)|(cat.PROB_GAL>prob_thres)]
        
        cat['class'] = 'galaxy'
        cat.loc[cat.CLASS == 6, 'class'] = 'star'

        cat['x_min'] = cat.X - m*cat.FWHM
        cat['x_max'] = cat.X + m*cat.FWHM
        cat['y_min'] = cat.Y - m*cat.FWHM
        cat['y_max'] = cat.Y + m*cat.FWHM

        reject = (np.floor(cat.x_min/patch_size) != np.floor(cat.x_max/patch_size)) | (
            np.floor(cat.y_min/patch_size) != np.floor(cat.y_max/patch_size))
        cat = cat[~reject]

        cat['image_x'] = np.floor(cat.x_min/patch_size).astype(int).astype(str)
        cat['image_y'] = np.floor(cat.y_min/patch_size).astype(int).astype(str)
        cat['image'] = cat[['image_y', 'image_x']].apply(
            lambda s: '{}{}.{}.{}.png'.format(output_folder, stripe, s[0], s[1]), axis=1)

        cat['x_min'] = cat.x_min % patch_size
        cat['x_max'] = cat.x_max % patch_size
        cat['y_min'] = cat.y_min % patch_size
        cat['y_max'] = cat.y_max % patch_size

        int_cols = ['x_min', 'x_max', 'y_min', 'y_max']
        cat[int_cols] = cat[int_cols].round(0).astype(int)

        zero_width = cat.x_min == cat.x_max
        cat.loc[zero_width, 'x_min'] = cat.x_min - 5
        cat.loc[zero_width, 'x_max'] = cat.x_max + 5
        zero_height = cat.y_min == cat.y_max
        cat.loc[zero_height, 'y_min'] = cat.y_min - 5
        cat.loc[zero_height, 'y_max'] = cat.y_max + 5

        cat_final = cat[['image', 'x_min', 'y_min', 'x_max', 'y_max', 'class']]
        cat_final.to_csv(cat_file, index=False, header=False, mode='a')


if __name__=='__main__':

    gen_data(in_path, out_path, cat_file)