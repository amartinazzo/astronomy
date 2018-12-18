from datetime import datetime
import json
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras_retinanet import losses
from keras_retinanet.models.resnet import resnet50_retinanet
from keras_retinanet.preprocessing.csv_generator import CSVGenerator
import numpy as np
import pandas as pd
import tensorflow as tf


cat_file = 'catalog_train.csv'
classes_file = 'class_mapping.csv'
pretrained_model = 'resnet50_coco_best_v2.1.0.h5'

train_file = 'catalog_train_train.csv'
val_file = 'catalog_train_val.csv'

lr = 1e-4
n_epochs = 10
val_split = 0.8

model_name = 'model-' + datetime.now().strftime("%y%m%d-%H%M")
checkpoint_file = '{}.h5'.format(model_name)


# HELPER FUNCS


def frozen_model(model):
    for layer in model.layers:
        layer.trainable = False
    return model


def load_model(weights, n_classes, freeze=True):
    modifier = frozen_model if freeze else None

    model = resnet50_retinanet(num_classes=n_classes, modifier=modifier)
    model.load_weights(weights, by_name=True, skip_mismatch=True)

    return model


# MAIN


if __name__=='__main__':

    # train val split

    df = pd.read_csv(
        cat_file, header=None, comment='#',
        names=['file', 'x0', 'y0', 'x1', 'y1', 'class'])
    df['split'] = np.random.randn(df.shape[0], 1)
    msk = np.random.rand(len(df)) <= val_split
    df.drop(columns=['split'], inplace=True)

    cols = ['x0', 'y0', 'x1', 'y1']
    df[cols] = df[cols].astype(int)
    df[msk].to_csv(train_file, index=False, header=False)
    df[~msk].to_csv(val_file, index=False, header=False)

    # load model

    model = load_model(pretrained_model, n_classes=2, freeze=True)

    # train model

    train_gen = CSVGenerator(train_file, classes_file)
    val_gen = CSVGenerator(val_file, classes_file)

    model.compile(
        loss={
            'regression'    : losses.smooth_l1(),
            'classification': losses.focal()
        },
        optimizer=optimizers.adam(lr=lr, clipnorm=0.001)
    )

    checkpoint = ModelCheckpoint(
        checkpoint_file,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=0
    )

    history = model.fit_generator(
        train_gen,
        steps_per_epoch=train_gen.size(),
        validation_data=val_gen,
        validation_steps=val_gen.size(),
        callbacks=[checkpoint],
        epochs=n_epochs
    )

    # save training history
    
    with open('{}.json'.format(model_name), 'w') as f:
    json.dump(history.history, f)