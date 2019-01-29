from constants import *
from train import load_model
from utils import non_max_suppression
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.models import convert_model
from keras_retinanet.utils.visualization import draw_box, draw_caption
import matplotlib.image as matplotim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

raw = False
predict = True
ground_truth = True

image_file = 'patches_t/STRIPE82-0013.1.9.png'
catalog = 'catalog_t.csv'
model_weights = 'models/model-190129.h5'

nms_thres = 0.9

# stars = red
# galaxies = green
colors = {'star': [255, 0, 0], 'galaxy': [0,255,0]}
colors_pred = {0: [255, 150, 150], 1: [150, 255, 150]}


image = read_image_bgr(image_file)

if ground_truth:
    cat = pd.read_csv(catalog, header=None, comment='#', names=df_cols)
    cat = cat[cat.file == image_file]

    int_cols = ['x0', 'x1', 'y0', 'y1']
    cat[int_cols] = cat[int_cols].astype(int)

    boxes = cat[['x0', 'y0', 'x1', 'y1']].values
    labels = cat['class'].values

    for box, label in zip(boxes, labels):
        draw_box(image, box, color=colors[label], style='dotted')

if predict:
    model = load_model(model_weights, n_classes=2)
    model = convert_model(model)
    image_input = preprocess_image(image)
    image_input, scale = resize_image(image_input)
    print('predicting...')
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(
        np.expand_dims(image_input, axis=0))
    print('processing time: ', time.time() - start)
    boxes /= scale

    print(labels[0])

    boxes_nms = non_max_suppression(boxes[0], nms_thres)

    for box, label, score in zip(boxes_nms, labels[0], scores[0]):
        if label == -1:
            break
        draw_box(image, box, color=colors_pred[label])
        draw_caption(image, box, '{:.2f}'.format(score))

plt.figure()
plt.axis('off')
plt.imshow(image)
matplotim.imsave('{}-{}-nms0.9-annotations.png'.format(
    image_file.split('/')[1][:-4],
    model_weights.split('/')[1][:-3]),
    image)
plt.show()
