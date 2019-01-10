import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='1' # use only gpu id=1

from datetime import datetime
import json
from keras import optimizers
from keras.callbacks import ModelCheckpoint, History
from keras_retinanet import losses
from keras_retinanet.models.resnet import resnet50_retinanet
from keras_retinanet.preprocessing.csv_generator import CSVGenerator
import numpy as np
import pandas as pd
import tensorflow as tf


# to keep training after logging out of server
# nohup python3 train.py &


do_train_val_split = False

cat_file = 'catalog_train.csv'
classes_file = 'class_mapping.csv'
pretrained_model = 'resnet50_coco_best_v2.1.0.h5'

train_file = 'catalog_train_train.csv'
val_file = 'catalog_train_val.csv'

clipnorm = 1e-3
freeze_weights = False
lr = 1e-4
n_epochs = 5
val_split = 0.3

model_name = 'model-' + datetime.now().strftime("%y%m%d-%H%M")
checkpoint_file = 'models/{}.h5'.format(model_name)


# HELPER FUNCS/CLASSES

# source at https://github.com/keras-team/keras/blob/master/keras/callbacks.py
class JsonHistory(History):
    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        with open('models/{}.json'.format(model_name), 'w') as file:
            json.dump(self.history, file)


def frozen_model(model):
    for layer in model.layers:
        layer.trainable = False
    return model


def load_model(weights, n_classes, freeze=True, anchor_params=None):
    modifier = frozen_model if freeze else None

    model = resnet50_retinanet(
        n_classes,
        modifier=modifier,
        anchor_parameters=anchor_params
        )
    model = model.load_weights(weights, by_name=True, skip_mismatch=True)

    return model


# MAIN


if __name__=='__main__':

    # train val split

    if do_train_val_split:
        df = pd.read_csv(
            cat_file, header=None, comment='#',
            names=['file', 'x0', 'y0', 'x1', 'y1', 'class'])
        df['split'] = np.random.randn(df.shape[0], 1)
        msk = np.random.rand(len(df)) >= val_split
        df.drop(columns=['split'], inplace=True)

        cols = ['x0', 'y0', 'x1', 'y1']
        df[cols] = df[cols].astype(int)
        df[msk].to_csv(train_file, index=False, header=False)
        df[~msk].to_csv(val_file, index=False, header=False)

    # load model

    model = load_model(
        pretrained_model,
        n_classes=2,
        freeze=freeze_weights,
        anchor_params=anchor_params
        )

    # train model

    train_gen = CSVGenerator(train_file, classes_file)
    val_gen = CSVGenerator(val_file, classes_file)

    model.compile(
        loss={
            'regression'    : losses.smooth_l1(),
            'classification': losses.focal()
        },
        metrics=['accuracy'],
        optimizer=optimizers.adam(lr=lr, clipnorm=clipnorm)
    )

    checkpoint = ModelCheckpoint(
        checkpoint_file,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=0
    )
    history_callback = JsonHistory()

    history = model.fit_generator(
        train_gen,
        steps_per_epoch=train_gen.size(),
        validation_data=val_gen,
        validation_steps=val_gen.size(),
        callbacks=[checkpoint, history_callback],
        epochs=n_epochs,
        verbose=1,
    )
