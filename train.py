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
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.utils import plot_model
from keras.layers import Dense, Dropout
from keras.models import Model
from sklearn.metrics import recall_score
import pandas as pd
import tables
from datetime import datetime
import argparse

def heartnet_transfer(load_path='/media/taufiq/Data/heart_sound/interspeech_compare/weights.0148-0.8902.hdf5',lr=0.0012843784,lr_decay=0.0001132885,num_dense=20,trainable=False):
    model = heartnet(load_path=load_path,FIR_train=False,trainable=trainable)
    x = model.layers[-4].output
    x = Dense(num_dense,activation='relu') (x)
    x = Dropout(rate=0.5,seed=1) (x)
    output = Dense(3,activation='softmax')(x)
    model = Model(inputs=model.input,outputs=output)
    sgd = SGD(lr=lr)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == '__main__':

    fold_dir = '/media/taufiq/Data/heart_sound/interspeech_compare/feature/segmented_noFIR/'
    foldname = 'comParE'
    model_dir = '/media/taufiq/Data/heart_sound/models/'
    log_name = foldname + ' ' + str(datetime.now())
    checkpoint_name = model_dir + log_name + "/" + 'weights.{epoch:04d}-{val_acc:.4f}.hdf5'
    ##### Load Model ######

    load_path='/media/taufiq/Data/heart_sound/interspeech_compare/weights.0148-0.8902.hdf5'
    lr = 0.00001
    num_dense = 20
    epochs = 100
    trainable = True
    model = heartnet_transfer(load_path=load_path,lr=lr,num_dense=num_dense,trainable=trainable)
    plot_model(model,"model.png",show_layer_names=True,show_shapes=True)

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

    ### Callbacks ###
    csv_logger = CSVLogger('training.csv')
    modelcheckpnt = ModelCheckpoint(filepath=checkpoint_name,
                                    monitor='val_acc', save_best_only=True, mode='max')

    ### Train ###
    model.fit(x_train,y_train,
              batch_size=128,
              epochs=epochs,
              verbose=2,
              shuffle=True,
              callbacks=[ModelCheckpoint, csv_logger],
              validation_data=(x_val,y_val))

    res_thresh =.5
    y_pred = model.predict(x_val, verbose=0)
    y_pred = np.argmax(y_pred,axis=-1)
    y_val = np.transpose(np.argmax(y_val,axis=-1))
    true = []
    pred = []
    start_idx = 0
    for s in val_parts:

        if not s:  ## for e00032 in validation0 there was no cardiac cycle
            continue
        # ~ print "part {} start {} stop {}".format(s,start_idx,start_idx+int(s)-1)

        temp_ = np.mean(y_val[start_idx:start_idx + int(s) - 1])
        temp = np.mean(y_pred[start_idx:start_idx + int(s) - 1])

        if temp > res_thresh:
            pred.append(1)
        else:
            pred.append(0)
        if temp_ > res_thresh:
            true.append(1)
        else:
            true.append(0)

        start_idx = start_idx + int(s)
    score = recall_score(true, pred, average='macro')
    print(score)
