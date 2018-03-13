from __future__ import print_function, division


from keras.constraints import max_norm
from keras.regularizers import l2

from custom_layers import Conv1D_linearphase
from heartnet_v1 import heartnet, reshape_folds, branch
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
from keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard, Callback
from keras.utils import plot_model
from keras.layers import Dense, Dropout, Concatenate, initializers, Input
from keras.models import Model
from sklearn.metrics import recall_score, confusion_matrix
import pandas as pd
import os
import tables
from datetime import datetime
import argparse
# def heartnet_transfer_raw(load_path,activation_function='relu', bn_momentum=0.99, bias=False, dropout_rate=0.5, dropout_rate_dense=0.0,
#              eps=1.1e-5, kernel_size=5, l2_reg=0.0, l2_reg_dense=0.0,lr=0.0012843784, lr_decay=0.0001132885, maxnorm=10000.,
#              padding='valid', random_seed=1, subsam=2, num_filt=(8, 4), num_dense=20,FIR_train=False,trainable=True):
#     input = Input(shape=(2500, 1))
#
#     coeff_path = 'filterbankcoeff60.mat'
#     coeff = tables.open_file(coeff_path)
#     b1 = coeff.root.b1[:]
#     b1 = np.hstack(b1)
#     b1 = np.reshape(b1, [b1.shape[0], 1, 1])
#
#     b2 = coeff.root.b2[:]
#     b2 = np.hstack(b2)
#     b2 = np.reshape(b2, [b2.shape[0], 1, 1])
#
#     b3 = coeff.root.b3[:]
#     b3 = np.hstack(b3)
#     b3 = np.reshape(b3, [b3.shape[0], 1, 1])
#
#     b4 = coeff.root.b4[:]
#     b4 = np.hstack(b4)
#     b4 = np.reshape(b4, [b4.shape[0], 1, 1])
#
#     input1 = Conv1D_linearphase(1, 61, use_bias=False,
#                                 # kernel_initializer=initializers.he_normal(random_seed),
#                                 weights=[b1[30:]],
#                                 padding='same', trainable=FIR_train)(input)
#     input2 = Conv1D_linearphase(1, 61, use_bias=False,
#                                 # kernel_initializer=initializers.he_normal(random_seed),
#                                 weights=[b2[30:]],
#                                 padding='same', trainable=FIR_train)(input)
#     input3 = Conv1D_linearphase(1, 61, use_bias=False,
#                                 # kernel_initializer=initializers.he_normal(random_seed),
#                                 weights=[b3[30:]],
#                                 padding='same', trainable=FIR_train)(input)
#     input4 = Conv1D_linearphase(1, 61, use_bias=False,
#                                 # kernel_initializer=initializers.he_normal(random_seed),
#                                 weights=[b4[30:]],
#                                 padding='same', trainable=FIR_train)(input)
#
#     t1 = branch(input1, num_filt, kernel_size, random_seed, padding, bias, maxnorm, l2_reg,
#                 eps, bn_momentum, activation_function, dropout_rate, subsam, trainable)
#     t2 = branch(input2, num_filt, kernel_size, random_seed, padding, bias, maxnorm, l2_reg,
#                 eps, bn_momentum, activation_function, dropout_rate, subsam, trainable)
#     t3 = branch(input3, num_filt, kernel_size, random_seed, padding, bias, maxnorm, l2_reg,
#                 eps, bn_momentum, activation_function, dropout_rate, subsam, trainable)
#     t4 = branch(input4, num_filt, kernel_size, random_seed, padding, bias, maxnorm, l2_reg,
#                 eps, bn_momentum, activation_function, dropout_rate, subsam, trainable)
#
#     # t1 = Conv1D(num_filt1, kernel_size=kernel_size,
#     #             kernel_initializer=initializers.he_normal(seed=random_seed),
#     #             padding=padding,
#     #             use_bias=bias,
#     #             kernel_constraint=max_norm(maxnorm),
#     #             kernel_regularizer=l2(l2_reg))(input1)
#     # t1 = BatchNormalization(epsilon=eps, momentum=bn_momentum, axis=-1)(t1)
#     # t1 = Activation(activation_function)(t1)
#     # t1 = Dropout(rate=dropout_rate, seed=random_seed)(t1)
#     # t1 = MaxPooling1D(pool_size=subsam)(t1)
#     # t1 = Conv1D(num_filt2, kernel_size=kernel_size,
#     #             kernel_initializer=initializers.he_normal(seed=random_seed),
#     #             padding=padding,
#     #             use_bias=bias,
#     #             kernel_constraint=max_norm(maxnorm),
#     #             kernel_regularizer=l2(l2_reg))(t1)
#     # t1 = BatchNormalization(epsilon=eps, momentum=bn_momentum, axis=-1)(t1)
#     # t1 = Activation(activation_function)(t1)
#     # t1 = Dropout(rate=dropout_rate, seed=random_seed)(t1)
#     # t1 = MaxPooling1D(pool_size=subsam)(t1)
#     # t1 = Flatten()(t1)
#
#     # t2 = Conv1D(num_filt1, kernel_size=kernel_size,
#     #             kernel_initializer=initializers.he_normal(seed=random_seed),
#     #             padding=padding,
#     #             use_bias=bias,
#     #             kernel_constraint=max_norm(maxnorm),
#     #             kernel_regularizer=l2(l2_reg))(input2)
#     # t2 = BatchNormalization(epsilon=eps, momentum=bn_momentum, axis=-1)(t2)
#     # t2 = Activation(activation_function)(t2)
#     # t2 = Dropout(rate=dropout_rate, seed=random_seed)(t2)
#     # t2 = MaxPooling1D(pool_size=subsam)(t2)
#     # t2 = Conv1D(num_filt2, kernel_size=kernel_size,
#     #             kernel_initializer=initializers.he_normal(seed=random_seed),
#     #             padding=padding,
#     #             use_bias=bias,
#     #             kernel_constraint=max_norm(maxnorm),
#     #             kernel_regularizer=l2(l2_reg))(t2)
#     # t2 = BatchNormalization(epsilon=eps, momentum=bn_momentum, axis=-1)(t2)
#     # t2 = Activation(activation_function)(t2)
#     # t2 = Dropout(rate=dropout_rate, seed=random_seed)(t2)
#     # t2 = MaxPooling1D(pool_size=subsam)(t2)
#     # t2 = Flatten()(t2)
#
#     # t3 = Conv1D(num_filt1, kernel_size=kernel_size,
#     #             kernel_initializer=initializers.he_normal(seed=random_seed),
#     #             padding=padding,
#     #             use_bias=bias,
#     #             kernel_constraint=max_norm(maxnorm),
#     #             kernel_regularizer=l2(l2_reg))(input3)
#     # t3 = BatchNormalization(epsilon=eps, momentum=bn_momentum, axis=-1)(t3)
#     # t3 = Activation(activation_function)(t3)
#     # t3 = Dropout(rate=dropout_rate, seed=random_seed)(t3)
#     # t3 = MaxPooling1D(pool_size=subsam)(t3)
#     # t3 = Conv1D(num_filt2, kernel_size=kernel_size,
#     #             kernel_initializer=initializers.he_normal(seed=random_seed),
#     #             padding=padding,
#     #             use_bias=bias,
#     #             kernel_constraint=max_norm(maxnorm),
#     #             kernel_regularizer=l2(l2_reg))(t3)
#     # t3 = BatchNormalization(epsilon=eps, momentum=bn_momentum, axis=-1)(t3)
#     # t3 = Activation(activation_function)(t3)
#     # t3 = Dropout(rate=dropout_rate, seed=random_seed)(t3)
#     # t3 = MaxPooling1D(pool_size=subsam)(t3)
#     # t3 = Flatten()(t3)
#
#     # t4 = Conv1D(num_filt1, kernel_size=kernel_size,
#     #             kernel_initializer=initializers.he_normal(seed=random_seed),
#     #             padding=padding,
#     #             use_bias=bias,
#     #             kernel_constraint=max_norm(maxnorm),
#     #             kernel_regularizer=l2(l2_reg))(input4)
#     # t4 = BatchNormalization(epsilon=eps, momentum=bn_momentum, axis=-1)(t4)
#     # t4 = Activation(activation_function)(t4)
#     # t4 = Dropout(rate=dropout_rate, seed=random_seed)(t4)
#     # t4 = MaxPooling1D(pool_size=subsam)(t4)
#     # t4 = Conv1D(num_filt2, kernel_size=kernel_size,
#     #             kernel_initializer=initializers.he_normal(seed=random_seed),
#     #             padding=padding,
#     #             use_bias=bias,
#     #             kernel_constraint=max_norm(maxnorm),
#     #             kernel_regularizer=l2(l2_reg))(t4)
#     # t4 = BatchNormalization(epsilon=eps, momentum=bn_momentum, axis=-1)(t4)
#     # t4 = Activation(activation_function)(t4)
#     # t4 = Dropout(rate=dropout_rate, seed=random_seed)(t4)
#     # t4 = MaxPooling1D(pool_size=subsam)(t4)
#     # t4 = Flatten()(t4)
#
#     merged = Concatenate(axis=1)([t1, t2, t3, t4])
#
#     merged = Dense(num_dense,
#                    activation=activation_function,
#                    kernel_initializer=initializers.he_normal(seed=random_seed),
#                    use_bias=bias,
#                    kernel_constraint=max_norm(maxnorm),
#                    kernel_regularizer=l2(l2_reg_dense))(merged)
#     # ~ merged = BatchNormalization(epsilon=eps,momentum=bn_momentum,axis=-1) (merged)
#     merged = Dropout(rate=dropout_rate_dense, seed=random_seed)(merged)
#     merged_ = Dense(2, activation='sigmoid')(merged)
#
#     model = Model(inputs=input, outputs=merged_)
#
#     if load_path:  # If path for loading model was specified
#         model.load_weights(filepath=load_path, by_name=False)
#
#     adam = Adam(lr=lr, decay=lr_decay)
#     model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
#
#     return model


def heartnet_transfer(load_path='/media/taufiq/Data/heart_sound/interspeech_compare/weights.0148-0.8902.hdf5',lr=0.0012843784,lr_decay=0.0001132885,num_dense1=20,num_dense2=20,trainable=False,dropout_rate=0.):
    model = heartnet(load_path=load_path,FIR_train=False,trainable=trainable)
    plot_model(model,'before.png',show_shapes=True,show_layer_names=True)
    x = model.layers[-4].output
    x = Dense(num_dense1,activation='relu') (x)
    x = Dropout(rate=dropout_rate,seed=1) (x)
    x = Dense(num_dense2, activation='relu')(x)
    x = Dropout(rate=dropout_rate, seed=1)(x)
    output = Dense(3,activation='softmax')(x)
    model = Model(inputs=model.input,outputs=output)
    plot_model(model, 'after.png',show_shapes=True,show_layer_names=True)
    if load_path:
        model.load_weights(load_path,by_name=True)
    sgd = SGD(lr=lr)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

class log_UAR(Callback):

    def __init__(self, x_val, y_val, val_parts):
        self.x_val = x_val
        self.y_val = y_val
        self.val_parts = val_parts


    def on_epoch_end(self, epoch, logs):
        if logs is not None:
            y_pred = self.model.predict(self.x_val, verbose=0)
            y_pred = np.argmax(y_pred, axis=-1)
            self.y_val_ = np.transpose(np.argmax(self.y_val, axis=-1))
            true = []
            pred = []
            start_idx = 0
            for s in self.val_parts:

                if not s:  ## for e00032 in validation0 there was no cardiac cycle
                    continue
                # ~ print "part {} start {} stop {}".format(s,start_idx,start_idx+int(s)-1)

                temp_ = self.y_val_[start_idx:start_idx + int(s) - 1]
                temp = y_pred[start_idx:start_idx + int(s) - 1]

                if (sum(temp == 0) > sum(temp == 1)) and (sum(temp == 0) > sum(temp == 2)):
                    pred.append(0)
                elif (sum(temp == 2) > sum(temp == 1)) and (sum(temp == 2) > sum(temp == 0)):
                    pred.append(2)
                else:
                    pred.append(1)

                if (sum(temp_ == 0) > sum(temp_ == 1)) and (sum(temp_ == 0) > sum(temp_ == 2)):
                    true.append(0)
                elif (sum(temp_ == 2) > sum(temp_ == 1)) and (sum(temp_ == 2) > sum(temp_ == 0)):
                    true.append(2)
                else:
                    true.append(1)

                start_idx = start_idx + int(s)

            score = recall_score(y_pred=pred, y_true=true, average='macro')
            logs['UAR'] = np.array(score)


if __name__ == '__main__':

    fold_dir = '/media/taufiq/Data/heart_sound/interspeech_compare/feature/segmented_noFIR/'
    foldname = 'comParE'
    model_dir = '/media/taufiq/Data/heart_sound/models/'
    log_name = foldname + ' ' + str(datetime.now())
    checkpoint_name = model_dir + log_name + "/" + 'weights.{epoch:04d}-{val_acc:.4f}.hdf5'
    if not os.path.exists(model_dir + log_name):
        os.makedirs(model_dir + log_name)
    log_dir = '/media/taufiq/Data/heart_sound/interspeech_compare/logs/'

    ##### Load Model ######

    load_path='/media/taufiq/Data/heart_sound/models/fold1_noFIR 2018-03-13 03:55:23.240321/weights.0169-0.8798.hdf5'
    # lr = 0.00001
    lr = 0.1
    num_dense1 = 650 #34,120,167,239,1239
    num_dense2 = 121 #121,
    epochs = 100
    batch_size = 256
    dropout_rate = 0.
    trainable = True
    # res_thresh = .5
    model = heartnet_transfer(load_path=load_path,lr=lr,num_dense1=num_dense1,num_dense2=num_dense2,trainable=trainable,dropout_rate=dropout_rate)
    plot_model(model,"model.png",show_layer_names=True,show_shapes=True)

    ###### Load Data ######

    feat = tables.open_file(fold_dir + foldname + '.mat')
    x_train = feat.root.trainX[:]
    y_train = feat.root.trainY[0, :]
    x_val = feat.root.valX[:]
    y_val = feat.root.valY[0, :]
    train_parts = feat.root.train_parts[:]
    val_parts = feat.root.val_parts[0, :]

    ############### Reshaping ############

    x_train, y_train, x_val, y_val = reshape_folds(x_train, x_val, y_train, y_val)
    y_train = to_categorical(y_train,num_classes=3)
    y_val = to_categorical(y_val,num_classes=3)

    ### Callbacks ###
    csv_logger = CSVLogger(log_dir + log_name + '/training.csv')
    modelcheckpnt = ModelCheckpoint(filepath=checkpoint_name,
                                    monitor='val_acc', save_best_only=True, mode='max')
    tensbd = TensorBoard(log_dir=log_dir + log_name,
                         batch_size=batch_size,
                         histogram_freq=100,
                         # embeddings_freq=99,
                         # embeddings_layer_names=embedding_layer_names,
                         # embeddings_data=x_val,
                         # embeddings_metadata=metadata_file,
                         write_images=False)
    print(np.sum(y_train,axis=-2))
    print(np.sum(y_val, axis=-2))
    ### Train ###
    model.fit(x_train,y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              shuffle=True,
              callbacks=[modelcheckpnt,
                         log_UAR(x_val, y_val, val_parts),
                         tensbd, csv_logger],
              validation_data=(x_val,y_val))

    ###### Results #####

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

        temp_ = y_val[start_idx:start_idx + int(s) - 1]
        temp = y_pred[start_idx:start_idx + int(s) - 1]

        if (sum(temp==0) > sum(temp==1)) and  (sum(temp==0) > sum(temp==2)):
            pred.append(0)
        elif (sum(temp==2) > sum(temp==1)) and  (sum(temp==2) > sum(temp==0)):
            pred.append(2)
        else:
            pred.append(1)

        if (sum(temp_ == 0) > sum(temp_ == 1)) and (sum(temp_ == 0) > sum(temp_ == 2)):
            true.append(0)
        elif (sum(temp_ == 2) > sum(temp_ == 1)) and (sum(temp_ == 2) > sum(temp_ == 0)):
            true.append(2)
        else:
            true.append(1)

        start_idx = start_idx + int(s)
    score = recall_score(y_pred=pred, y_true=true, average='macro')
    print(score)
    print(confusion_matrix(y_true=true,y_pred=pred))
