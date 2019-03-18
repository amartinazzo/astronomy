# input image		11000 x 11000
# original padding	im[1410:10560, 590:9980]
# squared padding	im[1100:10600, 480:9980]
# size with padding	9200 x 9500

# anchor params must be modified inside retinanet src code
# in ~/keras-retinanet/keras_retinanet/utils/anchors.py


pad_x0 = 1410
pad_x1 = 10560
pad_y0 = 590
pad_y1 = 9980


img_size = 11000
patch_overlap = 0
patch_size = 800		# retinanet will resize img if > 800

patch_step = patch_size - patch_overlap

m = 2				# multiply by FWHM to generate bounding boxes
f = 6				# min FWHM value

# thresholds for filtering training objects
fwhm_min = 1
fwhm_max = 20
mumax_min = 14
mumax_max = 17
prob_min = .9
s2n_min = 10
photoflag = 0

# patch_size + n*d = img_size
# n+1 is the number of patches along each axis

# CLASSES
star_int_class = 6
galaxy_int_class = 3

# MUMAX     Peak Surface Brightness above background [mag x arcsec^{-2}]
# s2nDet    Signal-to-Noise on Detection Image (FLUX_AUTO/FLUXERR_AUTO)

cols = [
    'ID', 'RA', 'Dec', 'X', 'Y', 'ISOarea', 's2nDet', 'PhotoFlag', 'FWHM', 'MUMAX', 'A', 'B', 'THETA', 'FlRadDet', 'KrRadDet', 'uJAVA_auto', 'euJAVA_auto', 's2n_uJAVA_auto', 'uJAVA_petro', 'euJAVA_petro', 's2n_uJAVA_petro', 'uJAVA_aper', 'euJAVA_aper', 's2n_uJAVA_aper', 'F378_auto', 'eF378_auto', 's2n_F378_auto', 'F378_petro', 'eF378_petro', 's2n_F378_petro', 'F378_aper', 'eF378_aper', 's2n_F378_aper', 'F395_auto', 'eF395_auto', 's2n_F395_auto', 'F395_petro', 'eF395_petro', 's2n_F395_petro', 'F395_aper', 'eF395_aper', 's2n_F395_aper', 'F410_auto', 'eF410_auto', 's2n_F410_auto', 'F410_petro', 'eF410_petro', 's2n_F410_petro', 'F410_aper', 'eF410_aper', 's2n_F410_aper', 'F430_auto', 'eF430_auto', 's2n_F430_auto', 'F430_petro', 'eF430_petro', 's2n_F430_petro', 'F430_aper', 'eF430_aper', 's2n_F430_aper', 'g_auto', 'eg_auto', 's2n_g_auto', 'g_petro', 'eg_petro', 's2n_g_petro', 'g_aper', 'eg_aper', 's2n_g_aper', 'F515_auto', 'eF515_auto', 's2n_F515_auto', 'F515_petro', 'eF515_petro', 's2n_F515_petro', 'F515_aper', 'eF515_aper', 's2n_F515_aper', 'r_auto', 'er_auto', 's2n_r_auto', 'r_petro', 'er_petro', 's2n_r_petro', 'r_aper', 'er_aper', 's2n_r_aper', 'F660_auto', 'eF660_auto', 's2n_F660_auto', 'F660_petro', 'eF660_petro', 's2n_F660_petro', 'F660_aper', 'eF660_aper', 's2n_F660_aper', 'i_auto', 'ei_auto', 's2n_i_auto', 'i_petro', 'ei_petro', 's2n_i_petro', 'i_aper', 'ei_aper', 's2n_i_aper', 'F861_auto', 'eF861_auto', 's2n_F861_auto', 'F861_petro', 'eF861_petro', 's2n_F861_petro', 'F861_aper', 'eF861_aper', 's2n_F861_aper', 'z_auto', 'ez_auto', 's2n_z_auto', 'z_petro', 'ez_petro', 's2n_z_petro', 'z_aper', 'ez_aper', 's2n_z_aper', 'zb', 'zb_Min', 'zb_Max', 'Tb', 'Odds', 'Chi2', 'M_B', 'Stell_Mass', 'CLASS', 'PROB_GAL', 'PROB_STAR'
]

df_cols = ['file', 'x0', 'y0', 'x1', 'y1', 'class']
usecols = ['ID', 'X', 'Y', 'MUMAX', 'PhotoFlag', 's2nDet', 'FWHM', 'CLASS', 'PROB_GAL', 'PROB_STAR']