import cv2
from glob import glob
import pandas as pd
import numpy as np
import os

# train or test
mode = 'train'

cat_file = 'catalog_{}.csv'.format(mode)
in_path = 'raw_data/'
out_path = 'patches_{}/'.format(mode)

patch_overlap = 200
patch_size = 1000
prob_thres = .9
m = 2               # multiply by FWHM to generate bounding boxes
f = 1               # if FWHM < f => FWHM = f

d = patch_overlap - patch_overlap

# CLASSES
# 3 galaxy
# 6 star

cols = [
    'ID', 'RA', 'Dec', 'X', 'Y', 'ISOarea', 's2nDet', 'PhotoFlag', 'FWHM', 'MUMAX', 'A', 'B', 'THETA', 'FlRadDet', 'KrRadDet', 'uJAVA_auto', 'euJAVA_auto', 's2n_uJAVA_auto', 'uJAVA_petro', 'euJAVA_petro', 's2n_uJAVA_petro', 'uJAVA_aper', 'euJAVA_aper', 's2n_uJAVA_aper', 'F378_auto', 'eF378_auto', 's2n_F378_auto', 'F378_petro', 'eF378_petro', 's2n_F378_petro', 'F378_aper', 'eF378_aper', 's2n_F378_aper', 'F395_auto', 'eF395_auto', 's2n_F395_auto', 'F395_petro', 'eF395_petro', 's2n_F395_petro', 'F395_aper', 'eF395_aper', 's2n_F395_aper', 'F410_auto', 'eF410_auto', 's2n_F410_auto', 'F410_petro', 'eF410_petro', 's2n_F410_petro', 'F410_aper', 'eF410_aper', 's2n_F410_aper', 'F430_auto', 'eF430_auto', 's2n_F430_auto', 'F430_petro', 'eF430_petro', 's2n_F430_petro', 'F430_aper', 'eF430_aper', 's2n_F430_aper', 'g_auto', 'eg_auto', 's2n_g_auto', 'g_petro', 'eg_petro', 's2n_g_petro', 'g_aper', 'eg_aper', 's2n_g_aper', 'F515_auto', 'eF515_auto', 's2n_F515_auto', 'F515_petro', 'eF515_petro', 's2n_F515_petro', 'F515_aper', 'eF515_aper', 's2n_F515_aper', 'r_auto', 'er_auto', 's2n_r_auto', 'r_petro', 'er_petro', 's2n_r_petro', 'r_aper', 'er_aper', 's2n_r_aper', 'F660_auto', 'eF660_auto', 's2n_F660_auto', 'F660_petro', 'eF660_petro', 's2n_F660_petro', 'F660_aper', 'eF660_aper', 's2n_F660_aper', 'i_auto', 'ei_auto', 's2n_i_auto', 'i_petro', 'ei_petro', 's2n_i_petro', 'i_aper', 'ei_aper', 's2n_i_aper', 'F861_auto', 'eF861_auto', 's2n_F861_auto', 'F861_petro', 'eF861_petro', 's2n_F861_petro', 'F861_aper', 'eF861_aper', 's2n_F861_aper', 'z_auto', 'ez_auto', 's2n_z_auto', 'z_petro', 'ez_petro', 's2n_z_petro', 'z_aper', 'ez_aper', 's2n_z_aper', 'zb', 'zb_Min', 'zb_Max', 'Tb', 'Odds', 'Chi2', 'M_B', 'Stell_Mass', 'CLASS', 'PROB_GAL', 'PROB_STAR'
]

usecols = ['ID', 'X', 'Y', 'FWHM', 'CLASS', 'PROB_GAL', 'PROB_STAR']


def patch_coord(x):
    if x <= patch_size:
        return 0
    return np.ceil((x - patch_size) / d)


def gen_data(input_folder, output_folder, csv_file):
    # if os.path.isdir(output_folder):
    #     print('error while generating data: output folder already exists')
    #     return 1
    # os.mkdir(output_folder)

    files = glob('{}{}_images/*'.format(input_folder, mode), recursive=True)
    n_files = len(files)

    for ix, filename in enumerate(files):
        print('{}/{} processing {}...'.format(ix+1, n_files, filename))
        stripe = filename.split('/')[-1].split('.')[0]

        # im = cv2.imread(filename)
        # imshape = im.shape
        # for y in range(0, imshape[0], d):
        #     y_int = y//d
        #     for x in range(0, imshape[1], d):
        #         cropped_img = im[y:y+patch_size, x:x+patch_size]
        #         cv2.imwrite('{}{}.{}.{}.png'.format(
        #             output_folder, stripe, y_int, x//d), cropped_img)

        cat = pd.read_csv('{}catalogs/SPLUS_{}_Photometry.cat'.format(input_folder, stripe),
            delimiter=' ', skipinitialspace=True, comment='#', index_col=False, header=None,
            names=cols, usecols=usecols)

        cat = cat[(cat.PROB_STAR>prob_thres)|(cat.PROB_GAL>prob_thres)]
        
        cat['class'] = 'galaxy'
        cat.loc[cat.CLASS == 6, 'class'] = 'star'

        cat.loc[cat.FWHM < f, 'FWHM'] = f
        cat['x0'] = cat.X - m*cat.FWHM# + cat.X//patch_size*patch_overlap
        cat['x1'] = cat.X + m*cat.FWHM# + cat.X//patch_size*patch_overlap
        cat['y0'] = cat.Y - m*cat.FWHM# + cat.Y//patch_size*patch_overlap
        cat['y1'] = cat.Y + m*cat.FWHM# + cat.Y//patch_size*patch_overlap

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

        zero_width = cat.x0 == cat.x1

        if cat[zero_width].shape[0] > 0:
            print(cat[zero_width].head())

        cat.loc[zero_width, 'x0'] = cat.x0 - f
        cat.loc[zero_width, 'x1'] = cat.x1 + f

        if cat[zero_width].shape[0] > 0:
            print(cat[zero_width].head())
            exit()

        zero_height = cat.y0 == cat.y1
        cat.loc[zero_height, 'y0'] = cat.y0 - f
        cat.loc[zero_height, 'y1'] = cat.y1 + f

        cat_final = cat[['image', 'x0', 'y0', 'x1', 'y1', 'class']]
        cat_final.to_csv(cat_file, index=False, header=False, mode='a')


if __name__=='__main__':
    gen_data(in_path, out_path, cat_file)