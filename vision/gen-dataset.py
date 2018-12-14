import cv2
from glob import glob
import pandas as pd
import numpy as np
import os

# train or test
mode = 'test'

cat_file = mode+'_catalog.csv'
in_path = 'raw_data/'
out_path = mode+'_patches/'

path_overlap = 200
patch_size = 1000
prob_thres = .9
m = 2               # multiply by FWHM to generate bounding boxes


def gen_data(input_folder, output_folder, csv_file):
    if os.path.isdir(output_folder):
        print('error while generating data: output folder already exists')
        return 1

    os.mkdir(output_folder)

    for filename in glob('{}{}_images/*'.format(input_folder, mode), recursive=True):
        stripe = filename.split('/')[-1].split('.')[0]

        im = cv2.imread(filename)
        imshape = im.shape

        d = patch_size - patch_overlap
        for y in range(0, imshape[0], d):
            y_int = y//d
            for x in range(0, imshape[1], d):
                cropped_img = im[y:y+patch_size, x:x+patch_size]
                cv2.imwrite('{}{}.{}.{}.png'.format(
                    output_folder, stripe, y_int, x//d), cropped_img)

        cat = pd.read_csv('{}catalogs/SPLUS_{}_Photometry.cat'.format(input_folder, stripe),
            delimiter=' ', skipinitialspace=True, comment='#', index_col=False)

        cat = cat[(cat.PROB_STAR>prob_thres)|(cat.PROB_GAL>prob_thres)]
        
        cat['class'] = 'galaxy'
        cat.loc[cat.CLASS == 6, 'class'] = 'star'

        cat['x0'] = cat.X - m*cat.FWHM
        cat['x1'] = cat.X + m*cat.FWHM
        cat['y0'] = cat.Y - m*cat.FWHM
        cat['y1'] = cat.Y + m*cat.FWHM

        reject = (np.floor(cat.x0/patch_size) != np.floor(cat.x1/patch_size)) | (
            np.floor(cat.y0/patch_size) != np.floor(cat.y1/patch_size))
        cat = cat[~reject]

        cat['image_x'] = np.floor(cat.x0/patch_size).astype(int).astype(str)
        cat['image_y'] = np.floor(cat.x0/patch_size).astype(int).astype(str)
        cat['image'] = cat[['image_y', 'image_x']].apply(
            lambda s: '{}{}.{}.{}.png'.format(output_folder, stripe, s[0], s[1]), axis=1)

        cat['x0'] = cat.x0 % patch_size
        cat['x1'] = cat.x1 % patch_size
        cat['y0'] = cat.y0 % patch_size
        cat['y1'] = cat.y1 % patch_size

        int_cols = ['x0', 'x1', 'y0', 'y1']
        cat[int_cols] = cat[int_cols].round(0).astype(int)

        zero_width = cat.x0 == cat.x1
        cat.loc[zero_width, 'x0'] = cat.x0 - 5
        cat.loc[zero_width, 'x1'] = cat.x1 + 5
        zero_height = cat.y0 == cat.y1
        cat.loc[zero_height, 'y0'] = cat.y0 - 5
        cat.loc[zero_height, 'y1'] = cat.y1 + 5

        cat_final = cat[['image', 'x0', 'y0', 'x1', 'y1', 'class']]
        cat_final.to_csv(cat_file, index=False, header=False, mode='a')


if __name__=='__main__':
    gen_data(in_path, out_path, cat_file)