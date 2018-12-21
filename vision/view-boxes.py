from constants import *
from keras_retinanet.utils.image import read_image_bgr
from keras_retinanet.utils.visualization import draw_box, draw_caption
import matplotlib.pyplot as plt
import pandas as pd

raw = False
image_file = 'patches_t/STRIPE82-0013.5.7.png'
catalog = 'catalog_t.csv'

colors = {'star': [255,0,0], 'galaxy': [0,255,0]}


if raw:
    cat = pd.read_csv(catalog,
        delimiter=' ', skipinitialspace=True, comment='#', index_col=False, header=None,
        names=cols, usecols=usecols)

    cat = cat[(cat.PROB_STAR>=prob_thres)|(cat.PROB_GAL>=prob_thres)]
    cat['class'] = 'galaxy'
    cat.loc[cat.CLASS == 6, 'class'] = 'star'
    cat.loc[cat.FWHM < f, 'FWHM'] = f
    cat['x0'] = cat.X - m*cat.FWHM
    cat['x1'] = cat.X + m*cat.FWHM
    cat['y0'] = y_size - cat.Y - m*cat.FWHM
    cat['y1'] = y_size - cat.Y + m*cat.FWHM

    cat = cat.sort_values(by='FWHM', ascending=False).head(100)
    #print(cat)

else:
    cat = pd.read_csv(catalog, header=None, comment='#', names=df_cols)
    cat = cat[cat.file == image_file]

int_cols = ['x0', 'x1', 'y0', 'y1']
cat[int_cols] = cat[int_cols].astype(int)

image = read_image_bgr(image_file)

boxes = cat[['x0', 'y0', 'x1', 'y1']].values
labels = cat['class'].values

print(boxes)

for box, label in zip(boxes, labels):
    draw_box(image, box, color=colors[label])
    #draw_caption(image, box, label)

plt.figure()
plt.axis('off')
plt.imshow(image)
plt.show()