# Thư viện hệ thống và xử lý dữ liệu
import os
from glob import glob
from pathlib import Path
import pandas as pd
import numpy as np
import gc
import pickle

# Thư viện học sâu với TensorFlow (Keras đã tích hợp sẵn)
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers
from tensorflow.keras.layers import (
    Dense, Activation, Flatten, Dropout, BatchNormalization, 
    Conv2D, MaxPooling2D, MaxPool2D, LeakyReLU
)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K

# Thư viện xử lý âm thanh và trực quan hóa
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib import figure


train_data_path='data/train/'
test_data_path='data/test/'
wav_path = 'data/wav/'

# Ham doi duoi tu wav sang png
def append_ext(fn):
    return fn.replace(".wav",".png")

# Load du lieu train va test tu file csv
traindf=pd.read_csv('data/train.csv',dtype=str)
testdf=pd.read_csv('data/test.csv',dtype=str)
traindf["slice_file_name"]=traindf["slice_file_name"].apply(append_ext)
testdf["slice_file_name"]=testdf["slice_file_name"].apply(append_ext)

datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.25)

# Load generator train
train_generator=datagen.flow_from_dataframe(
    dataframe=traindf,
    directory=train_data_path,
    x_col="slice_file_name",
    y_col="class",
    subset="training",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(64,64))

# Load generator val
valid_generator=datagen.flow_from_dataframe(
    dataframe=traindf,
    directory=train_data_path,
    x_col="slice_file_name",
    y_col="class",
    subset="validation",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(64,64))
'''
# Load generator train
train_generator_vgg=datagen.flow_from_dataframe(
    dataframe=traindf,
    directory=train_data_path,
    x_col="slice_file_name",
    y_col="class",
    subset="training",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(224,224))

# Load generator val
valid_generator_vgg=datagen.flow_from_dataframe(
    dataframe=traindf,
    directory=train_data_path,
    x_col="slice_file_name",
    y_col="class",
    subset="validation",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(224,224))
'''
# Khoi tao model
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(64,64,3)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(
    optimizer=RMSprop(learning_rate=0.0005),  # Sử dụng learning_rate thay vì lr
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

'''
# model VGG 16
modelvgg = Sequential()
modelvgg.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
modelvgg.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
modelvgg.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
modelvgg.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
modelvgg.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
modelvgg.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
modelvgg.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
modelvgg.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
modelvgg.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
modelvgg.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
modelvgg.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
modelvgg.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
modelvgg.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
modelvgg.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
modelvgg.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
modelvgg.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
modelvgg.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
modelvgg.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
modelvgg.add(Dropout(0.5))
modelvgg.add(Flatten())
modelvgg.add(Dense(units=4096,activation="relu"))
modelvgg.add(Dropout(0.5))
modelvgg.add(Dense(units=4096,activation="relu"))
modelvgg.add(Dropout(0.5))
modelvgg.add(Dense(units=10, activation="softmax"))
from keras.optimizers import Adam
opt = Adam(lr=0.001)
modelvgg.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
modelvgg.summary()
'''
# Tinh so buoc trong 1 epoch khi train
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
# Tinh so buoc trong 1 epoch khi val
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

# Train model
# Train model
model.fit(train_generator,
          steps_per_epoch=STEP_SIZE_TRAIN,
          validation_data=valid_generator,
          validation_steps=STEP_SIZE_VALID,
          epochs=100, verbose=1)



model.save("model.h5");
# Luu ten class
#np.save('model_indices', train_generator.class_indices)
with open('model_indices.pickle', 'wb') as handle:
    pickle.dump(train_generator.class_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Train model
'''modelvgg.fit_generator(generator=train_generator_vgg,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator_vgg,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=100,verbose=1
)'''

# Luu model
#modelvgg.save("modelvgg.h5");



print("Model trained!")
