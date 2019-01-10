from constants import *
from train import load_model
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.models import convert_model
from keras_retinanet.utils.visualization import draw_box, draw_caption
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

raw = False
predict = False

image_file = 'patches_t/STRIPE82-0013.0.12.png'
catalog = 'catalog_t.csv'
model_weights = 'models/model-190109-1157.h5'

# stars = red
# galaxies = blue
colors = {'star': [200,0,0], 'galaxy': [0,200,0]}
colors_pred = {'star': [255, 150, 150], 'galaxy': [150,255,150]}

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
    cat['y0'] = img_size - cat.Y - m*cat.FWHM
    cat['y1'] = img_size - cat.Y + m*cat.FWHM

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

for box, label in zip(boxes, labels):
    draw_box(image, box, color=colors[label])
    #draw_caption(image, box, label)

if predict:
    model = load_model(model_weights, n_classes=2, anchor_params=anchor_params)
    model = convert_model(model)
    image_input = preprocess_image(image)
    image_input, scale = resize_image(image_input)
    print('predicting...')
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image_input, axis=0))
    print('processing time: ', time.time() - start)
    boxes /= scale

    print(boxes[0])
    print(labels[0])

    for box, label in zip(boxes[0], labels[0]):
        if label == -1:
            break
        draw_box(image, box, color=colors_pred[label])

plt.figure()
plt.axis('off')
plt.imshow(image)
plt.show()