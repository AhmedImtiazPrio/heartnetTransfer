from __future__ import print_function, division
from heartnet_v1 import heartnet, reshape_folds
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))
import numpy as np
np.random.seed(1)
from tensorflow import set_random_seed
set_random_seed(1)
from keras.utils import to_categorical
from keras.optimizers import Adam, SGD
from keras.callbacks import CSVLogger
from keras.utils import plot_model
from keras.layers import Dense
from keras.models import Model

import pandas as pd
import tables
from datetime import datetime
import argparse

def heartnet_transfer(load_path='/media/taufiq/Data/heart_sound/interspeech_compare/weights.0148-0.8902.hdf5',lr=0.0012843784,lr_decay=0.0001132885):
    model = heartnet(load_path=load_path,FIR_train=False,trainable=False)
    x = model.layers[-2].output
    output = Dense(3,activation='softmax')(x)
    model = Model(inputs=model.input,outputs=output)
    sgd = SGD(lr=lr)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == '__main__':

    fold_dir = '/media/taufiq/Data/heart_sound/interspeech_compare/feature/segmented_noFIR/'
    foldname = 'comParE'

    ##### Load Model ######

    load_path='/media/taufiq/Data/heart_sound/interspeech_compare/weights.0148-0.8902.hdf5'
    lr = 0.0012843784
    model = heartnet_transfer(load_path=load_path,lr=lr)
    plot_model(model,"model.png")

    ##### Load Data ######

    feat = tables.open_file(fold_dir + foldname + '.mat')
    x_train = feat.root.trainX[:]
    y_train = feat.root.trainY[0, :]
    x_val = feat.root.valX[:]
    y_val = feat.root.valY[0, :]
    train_parts = feat.root.train_parts[:]
    val_parts = feat.root.val_parts[0, :]

    ################### Reshaping ############

    x_train, y_train, x_val, y_val = reshape_folds(x_train, x_val, y_train, y_val)
    y_train = to_categorical(y_train,num_classes=3)
    y_val = to_categorical(y_val,num_classes=3)
    csv_logger = CSVLogger('training.csv')
    model.fit(x_train,y_train,
              batch_size=128,
              epochs=20,
              shuffle=True,
              callbacks=[csv_logger],
              validation_data=(x_val,y_val))
