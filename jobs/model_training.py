from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import helper
import pickle

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D,Conv1D,MaxPooling1D

if tf.test.is_gpu_available():
    print('---- We are using GPU now ----')
    # GPU
    BATCH_SIZE = 512  # Number of examples used in each iteration
    EPOCHS = 800  # Number of passes through entire dataset

# Hyperparams for CPU training
else:
    print('---- We are using CPU now ----')
    # CPU
    BATCH_SIZE = 256
    EPOCHS = 100

print('---- Loading Data ----')
data_path = '/floyd/input/gengduoshuju/'  # ADD path/to/dataset
Y= pickle.load( open(os.path.join(data_path,'Y.pks'), "rb" ) )
X= pickle.load( open(os.path.join(data_path,'X.pks'), "rb" ) )
X = X.reshape((X.shape[0],X.shape[1],1))
print("Size of X :" + str(X.shape))
print("Size of Y :" + str(Y.shape))
X = X.astype(np.float64)
X = np.nan_to_num(X)

print('---- Split Data ----')
X_train,  X_test, Y_train_orig,Y_test_orig= helper.divide_data(X,Y)
print(Y.min())
print(Y.max())
num_classes = 332
Y_train = keras.utils.to_categorical(Y_train_orig, num_classes)
Y_test = keras.utils.to_categorical(Y_test_orig, num_classes)
print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))

print('---- Define model ----')
model = Sequential()
model.add(Conv1D(16, 4,padding='same',input_shape=X_train.shape[1:]))
print(model.output_shape)
model.add(Activation('relu'))
print(model.output_shape)
model.add(MaxPooling1D(2,padding='same'))
print(model.output_shape)
model.add(Conv1D(32, 4,padding='same'))
print(model.output_shape)
model.add(Activation('relu'))
model.add(MaxPooling1D(2,padding='same'))
print(model.output_shape)
model.add(Conv1D(64, 4,padding='same'))
print(model.output_shape)
model.add(Activation('relu'))
model.add(MaxPooling1D(2,padding='same'))
model.add(Conv1D(64, 4,padding='same'))
print(model.output_shape)
model.add(Activation('relu'))
model.add(MaxPooling1D(2,padding='same'))
model.add(Conv1D(32, 4,padding='same'))
print(model.output_shape)
model.add(Activation('relu'))
model.add(MaxPooling1D(2,padding='same'))
print(model.output_shape)
model.add(Flatten())
model.add(Dense(num_classes))
model.add(Activation('softmax'))

opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.summary()
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

history_try = model.fit(X_train, Y_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              validation_data=(X_test, Y_test),
              shuffle=True)

model.save(r"floyd_model_xxl_data_v.h5")
print('Training is done!')


# Plot training & validation accuracy values
with open('training_hist.pks','wb') as f:
    pickle.dump(history_try.history, f)

print('---- We are done with everything ----')