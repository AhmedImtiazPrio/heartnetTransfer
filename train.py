from __future__ import print_function, division
from heartnet_v1 import heartnet
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))
import numpy as np
np.random.seed(1)
from tensorflow import set_random_seed
set_random_seed(1)

from keras.optimizers import Adam, SGD

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
    model = heartnet_transfer()
    plot_model(model,"model.png")
