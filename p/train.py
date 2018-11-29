from collections import Counter
from datetime import datetime
from imblearn.over_sampling import SMOTE
import keras
from keras import backend as K
from keras import layers
from keras.models import Sequential, save_model
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import sys
import tensorflow as tf
import time

# PARAMS

output = False #write output to txt

batch_size = 256
epochs = 3000
eps = 1e-8
lr = 1e-4
n_neurons = 16
seed = 42

# TODO add a boolean argument to save image plots during training
# so that they can be opened through sshfs

timestamp = datetime.now().strftime("%m%d-%H%M")
checkpoint_file = 'models/model-{}.h5'.format(timestamp)
txt_file = 'output-{}.txt'.format(timestamp)

training_file = 'data/training_set.csv'
training_meta_file = 'data/training_set_metadata.csv'
test_file = 'data/test_set.csv'
test_meta_file = 'data/test_set_metadata.csv'


# features per light curve

# number of maxima
# number of minima
# minimum flux (flux - flux_err)
# maximum flux (flux + flux_err)
# global minima (min of minimum flux)
# global maxima (max of maximum flux)
# maximum flux sum
# minimum flux sum -> area??


# HELPER FUNCTIONS


def same_timeseries(df):
    return (df.object_id==df.object_id.shift(1)) & (df.object_id==df.object_id.shift(-1)) & (df.passband==df.passband.shift(1)) & (df.passband==df.passband.shift(-1))


def is_minima(df, col):
    s = (df.same_serie) & (df[col]<df[col].shift(1)) & (df[col]<df[col].shift(-1))
    return s.astype(int)


def is_maxima(df, col):
    s = (df.same_serie) & (df[col]>df[col].shift(1)) & (df[col]>df[col].shift(-1))
    return s.astype(int)


def pre_process_df(df, df_metadata, mode='test'):
    if type(df) is str:
        df = pd.read_csv(df)
    
    if type(df_metadata) is str:
        df_metadata = pd.read_csv(df_metadata)

    df.sort_values(['object_id', 'passband', 'mjd'], inplace=True)
    
    #diff_timeseries = ((df['object_id']!=df['object_id'].shift(1)) | (df['passband']!=df['passband'].shift(1)))
    #df.loc[diff_timeseries, 'timestep'] = df['timestep'].shift(-1)    

    n_passbands = df.passband.unique().shape[0]
    df['passband'] = df.passband.apply(str)
    
    df['same_serie'] = same_timeseries(df)
    df['flux_min'] = df.flux - df.flux_err
    df['flux_max'] = df.flux + df.flux_err
    df['minima'] = is_minima(df, 'flux_min')
    df['maxima'] = is_maxima(df, 'flux_max')

    agg_funcs = {
        'mjd': ['size'],
        'minima': ['mean'], #count/n_points
        'maxima': ['mean'], #count/n_points
        'flux_min': ['sum', 'min'],
        'flux_max': ['sum', 'max'],
        'flux': ['sum', 'mean', 'median', 'std'],
        'detected': ['mean'],
    }
    

    df = pd.pivot_table(
        df,
        values=['mjd', 'minima', 'maxima',  'flux_min', 'flux_max', 'flux', 'detected'],
        index=['object_id'],
        columns = ['passband'],
        aggfunc=agg_funcs
    )
    
    df.reset_index(inplace=True)
    df.columns = ['_'.join(col).strip() for col in df.columns.values]
    df.rename(columns={'object_id__': 'object_id'}, inplace=True)

    df = df.merge(
        right=df_metadata,
        how='left',
        on='object_id'
    )
    
    for p in range(n_passbands):
        df['amplitude_{}'.format(p)] = df['flux_max_max_{}'.format(p)] - df['flux_min_min_{}'.format(p)]
    
    object_ids = None

    if mode=='test':
        object_ids = df['object_id']

    
    drop_cols = ['object_id', 'ra', 'decl', 'gal_l', 'gal_b', 'ddf', 'distmod', 'mwebv', 'hostgal_photoz', 'hostgal_photoz_err']
    df.drop(columns=drop_cols, inplace=True)
    
    df.fillna(df.mean(axis=0), inplace=True)
    
    return df, object_ids


def get_input(df):
    if 'target' in df.columns:
        X = df.drop(columns=['target'])
        X = X.values
    else:
        X = df.values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X


def one_hot(y):
    return pd.get_dummies(pd.Series(y)).values


def wloss(y_true, y_pred):
    yc = tf.clip_by_value(y_pred,1e-15,1-1e-15)
    loss = -(tf.reduce_mean(tf.reduce_mean(y_true*tf.log(yc),axis=0)/wtable))
    return loss


def plot_loss(history):
    plt.plot(history.history['loss'][1:])
    plt.plot(history.history['val_loss'][1:])
    plt.title('model loss')
    plt.ylim(0,2)
    plt.ylabel('val_loss')
    plt.xlabel('epoch')
    plt.legend(['train','validation'], loc='upper left')
    plt.show()
    # TODO conditional to show or save plot


def build_model(n_features, n_classes, n_neurons=n_neurons, dropout_rate=0.5, activation='relu'):
    model = Sequential()
    model.add(layers.Dense(n_neurons, input_dim=n_features, activation=activation))
    model.add(layers.Dropout(dropout_rate))
    
    model.add(layers.Dense(n_classes, activation='softmax'))
    model.compile(loss=wloss, optimizer=Adam(lr=lr, epsilon=eps))
    
    return model


# GLOBAL VARIABLES


df, _ = pre_process_df(training_file, training_meta_file, mode='train')
X = get_input(df)
y = df['target'].values

# https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69795

unique_y = np.unique(y)
n_classes = len(unique_y)
n_features = X.shape[1]

class_map = dict()
for i,val in enumerate(unique_y):
    class_map[val] = i        
y_map = np.zeros((y.shape[0],))
y_map = np.array([class_map[val] for val in y])
y_count = Counter(y_map)

wtable = np.zeros((n_classes,))
for i in range(n_classes):
    wtable[i] = y_count[i]/y_map.shape[0]


# MAIN


if __name__=='main':

	# print to file

	if output:
	    orig_stdout = sys.stdout
	    f = open(txt_file, 'w')
	    sys.stdout = f

	print('\n\nHYPERPARAMS')
	print('batch_size={}\nepochs={}\nlr={}\nn_neurons={}\nseed={}'.format(batch_size, epochs, lr, n_neurons, seed))
	print('\n\nfull sets')
	print(X.shape)
	print(y.shape)

	# train test split

	splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
	train_idx, test_idx = list(splitter.split(X, y))[0]

	X_train, X_test = X[train_idx], X[test_idx]
	y_train, y_test = y[train_idx], y[test_idx]

	train_idx, val_idx = list(splitter.split(X_train, y_train))[0]
	X_train, X_val = X_train[train_idx], X_train[val_idx]
	y_train, y_val = y_train[train_idx], y_train[val_idx]

	print('\n\ntrain sets after train-test split')
	print(X_train.shape)
	print(y_train.shape)

	# class proportions

	unique, counts = np.unique(y_train, return_counts=True)
	counts = np.round(counts/len(y_train), 2)
	print('\n\nclass proportions')
	print(list(zip(unique,counts)))

	# one-hot encoding of classes

	# TODO use to_categorical func

	y_train = one_hot(y_train)
	y_val = one_hot(y_val)
	y_test = one_hot(y_test)
	
	# oversampling - only on the training set!
	# balanced classes: 7% each

	sm = SMOTE(random_state=seed)
	X_train, y_train = sm.fit_sample(X_train, y_train)

	unique, counts = np.unique(y_train, return_counts=True)
	counts = np.round(counts/len(y_train),2)
	print('\n\nnclass proportions after oversampling')
	print(list(zip(unique,counts)))

	# model training

	K.clear_session()
	model = build_model(n_features=n_features, n_classes=n_classes)
	print(model.summary())

	# checkpoint: save epoch with lowest validation loss
	checkpoint = keras.callbacks.ModelCheckpoint(
	        checkpoint_file, monitor='val_loss', mode='min', save_best_only=True, verbose=0)

	history = model.fit(
	    X_train, y_train,
	    epochs=epochs,
	    batch_size=batch_size,
	    validation_data=(X_val, y_val),
	    #validation_split=0.1,
	    verbose=2,
	    shuffle=True,
	    callbacks=[checkpoint]
	)

	#plot_loss(history)

	print('\n\nevaluation on test set')
	print(model.evaluate(X_test, y_test))