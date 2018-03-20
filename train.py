from __future__ import print_function, division
from keras.constraints import max_norm
from keras.regularizers import l2

from heartnet_v1 import heartnet, reshape_folds
import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
# set_session(tf.Session(config=config))
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
def compute_weight(Y, classes):
    num_samples = (len(Y))
    n_classes = (len(classes))
    num_bin = np.sum(Y,axis=-2)
    class_weights = {i: (num_samples / (n_classes * num_bin[i])) for i in range(n_classes)}
    return class_weights

def heartnet_transfer(load_path='/media/taufiq/Data/heart_sound/interspeech_compare/weights.0148-0.8902.hdf5',lr=0.0012843784,lr_decay=0.0001132885,num_dense1=20,num_dense2=20,trainable=False,dropout_rate=0.,optimizer=Adam):
    model = heartnet(load_path=False,FIR_train=False,trainable=trainable)
    plot_model(model,'before.png',show_shapes=True,show_layer_names=True)
    x = model.layers[-4].output
    x = Dense(num_dense1,activation='relu',kernel_initializer=initializers.he_uniform(seed=1)) (x)
    x = Dropout(rate=dropout_rate,seed=1) (x)
    x = Dense(num_dense2, activation='relu',kernel_initializer=initializers.he_normal(seed=1))(x)
    x = Dropout(rate=dropout_rate, seed=1)(x)
    output = Dense(3,activation='softmax')(x)
    model = Model(inputs=model.input,outputs=output)
    plot_model(model, 'after.png',show_shapes=True,show_layer_names=True)
    if load_path:
        model.load_weights(load_path,by_name=True)
    gd = optimizer(lr=lr)
    model.compile(optimizer=gd, loss='categorical_crossentropy', metrics=['accuracy'])
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

            # score = recall_score(y_pred=pred, y_true=true, average='macro')
            # logs['UAR'] = np.array(score)
            confmat = confusion_matrix(y_pred=pred, y_true=true)
            logs['recall0'] = confmat[0,0]/np.sum(confmat[0,:])
            logs['recall1'] = confmat[1,1]/np.sum(confmat[1,:])
            logs['recall2'] = confmat[2,2]/np.sum(confmat[2,:])
            logs['UAR'] = np.mean([logs['recall0'],logs['recall1'],logs['recall2']])


if __name__ == '__main__':

    fold_dir = '/home/prio/heart_sound/feature/segmented_noFIR/'
    # foldname = 'compare+valid1-normal+severe'
    foldname = 'compare+normalboost+severe'
    model_dir = '/home/prio/heart_sound/models/'
    log_name = foldname + ' ' + str(datetime.now())
    checkpoint_name = model_dir + log_name + "/" + 'weights.{epoch:04d}-{val_acc:.4f}.hdf5'
    if not os.path.exists(model_dir + log_name):
        os.makedirs(model_dir + log_name)
    log_dir = '/home/prio/heart_sound/logs/'
    results_path = '/home/prio/heart_sound/results.csv'

    ##### Load Model ######

    load_path='/home/prio/heart_sound/weights.0148-0.8902.hdf5'
    # lr = 0.00001
    lr = 0.006028585143146318
    # lr = 1e-5
    num_dense1 = 239 #34,120,167,239,1239,650,788,422,598,1458
    num_dense2 = 63 #121,
    epochs = 250
    batch_size = 512
    dropout_rate = 0.11366701825201375
    trainable = True
    addweights = False
    optimizer = SGD

    # res_thresh = .5
    model = heartnet_transfer(load_path=load_path,lr=lr,num_dense1=num_dense1,num_dense2=num_dense2,trainable=trainable,dropout_rate=dropout_rate,optimizer=optimizer)
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
    class_weights=compute_weight(y_train,range(3))
    print(class_weights)
    try:
        if addweights:
            model.fit(x_train,y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=2,
                  shuffle=True,
                  class_weight=class_weights,
                  callbacks=[modelcheckpnt,
                             log_UAR(x_val, y_val, val_parts),
                             tensbd, csv_logger],
                  validation_data=(x_val,y_val))
        else:
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

    ##### log results #####

        df = pd.read_csv(results_path)
        df1 = pd.read_csv(log_dir + log_name + '/training.csv')
        max_idx = df1['UAR'].idxmax()
        new_entry = {'Filename': log_name, 'Weight Initialization': 'he_uniform',
                     'Activation': 'softmax', 'Class weights': addweights,
                     'Learning Rate' : lr,
                     'Num Dense 1': num_dense1,
                     'Num Dense 2': num_dense2,
                     'Dropout rate': dropout_rate,
                     'l2_reg': 0.00,
                     'Optimizer': 'adam',
                     'Val Acc Per Cardiac Cycle': np.mean(df1.loc[max_idx - 3:max_idx + 3]['val_acc'].values) * 100,
                     'Val loss Per Cardiac Cycle' : np.mean(df1.loc[max_idx - 3:max_idx + 3]['val_loss'].values),
                     'Epoch': df1.loc[[max_idx]]['epoch'].values[0],
                     'Training Acc per cardiac cycle': np.mean(df1.loc[max_idx - 3:max_idx + 3]['acc'].values) * 100,
                     'Normal Recall' : np.mean(df1.loc[max_idx - 3:max_idx + 3]['recall0'].values) * 100,
                     'Mild Recall' : np.mean(df1.loc[max_idx - 3:max_idx + 3]['recall1'].values) * 100,
                     'Severe Recall' : np.mean(df1.loc[max_idx - 3:max_idx + 3]['recall2'].values) * 100,
                     'UAR': np.mean(df1.loc[max_idx - 3:max_idx + 3]['UAR'].values) * 100,
                     }

        index, _ = df.shape
        new_entry = pd.DataFrame(new_entry, index=[index])
        df2 = pd.concat([df, new_entry], axis=0)
        df2 = df2.reindex(df.columns, axis=1)
        df2.to_csv(results_path, index=False)
        df2.tail()

    except KeyboardInterrupt:
        df = pd.read_csv(results_path)
        df1 = pd.read_csv(log_dir + log_name + '/training.csv')
        max_idx = df1['UAR'].idxmax()
        new_entry = {'Filename': log_name, 'Weight Initialization': 'he_uniform',
                     'Activation': 'softmax', 'Class weights': addweights,
                     'Learning Rate': lr,
                     'Num Dense 1': num_dense1,
                     'Num Dense 2': num_dense2,
                     'Dropout rate': dropout_rate,
                     'l2_reg': 0.00,
                     'Optimizer': 'adam',
                     'Val Acc Per Cardiac Cycle': np.mean(df1.loc[max_idx - 3:max_idx + 3]['val_acc'].values) * 100,
                     'Val loss Per Cardiac Cycle': np.mean(df1.loc[max_idx - 3:max_idx + 3]['val_loss'].values),
                     'Epoch': df1.loc[[max_idx]]['epoch'].values[0],
                     'Training Acc per cardiac cycle': np.mean(df1.loc[max_idx - 3:max_idx + 3]['acc'].values) * 100,
                     'Normal Recall': np.mean(df1.loc[max_idx - 3:max_idx + 3]['recall0'].values) * 100,
                     'Mild Recall': np.mean(df1.loc[max_idx - 3:max_idx + 3]['recall1'].values) * 100,
                     'Severe Recall': np.mean(df1.loc[max_idx - 3:max_idx + 3]['recall2'].values) * 100,
                     'UAR': np.mean(df1.loc[max_idx - 3:max_idx + 3]['UAR'].values) * 100,
                     }

        index, _ = df.shape
        new_entry = pd.DataFrame(new_entry, index=[index])
        df2 = pd.concat([df, new_entry], axis=0)
        df2 = df2.reindex(df.columns, axis=1)
        df2.to_csv(results_path, index=False)
        df2.tail()