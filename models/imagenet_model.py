import os
import sys

import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.python.platform import flags
from keras import regularizers

FLAGS = flags.FLAGS

import pickle
from keras.models import Sequential, load_model, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Conv2D, GlobalAveragePooling2D
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.noise import GaussianNoise
from keras.layers.normalization import BatchNormalization
from keras.applications.vgg16 import VGG16
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet_v2 import MobileNetV2
import random
from scipy.misc import imread
from scipy.misc import imresize


class IMAGENET_model:
    def __init__(self, input_shape=(None, 32, 32, 3), nb_filters=64, nb_classes=10):
        self.input_shape = input_shape
        self.nb_filters = nb_filters
        self.nb_classes = nb_classes
        self.model = self.get_model()

    def get_model(self):
        '''
        Define the VGG16 structure
        :return:
        '''
        model = MobileNet(weights=None, classes=10)
        # model = MobileNetV2(weights=None, classes=10)
        return model

    def load_train_images(self, batch_shape, image_list):
        images = np.zeros(batch_shape)
        labels = np.zeros([batch_shape[0], 10])
        idx = 0
        batch_size = batch_shape[0]
        while (True):
            for file, label in image_list:
                image = imread(file, mode='RGB')
                images[idx, :, :, :] = image
                labels[idx, label] = 1
                idx += 1
                if idx == batch_size:
                    yield images / 255., labels
                    images = np.zeros(batch_shape)
                    labels = np.zeros([batch_shape[0], 10])
                    idx = 0
            if idx > 0:
                yield images / 255., labels
    def train(self, x_train, y_train, x_test, y_test, batch_size=128, nb_epochs=250, is_train=True):
        """
        detect adversarial examples
        :param x_train: train data
        :param y_train: train labels
        :param x_test:  test data
        :param y_test: test labels
        :param batch_size: batch size during training
        :param nb_epochs: number of iterations of model
        :param is_train: train online or load weight from file
        :return
        """
        batch_size = 64
        x_test_temp = x_test
        optimizer = Adam(lr=1e-3)  # Using Adam instead of SGD to speed up training
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
        generator = ImageDataGenerator(rotation_range=15,
                                       rescale=1. / 255,
                                       width_shift_range=0.15,
                                       height_shift_range=0.15,
                                       shear_range=0.15,
                                       zoom_range=0.15,
                                       horizontal_flip=True)
        train_generator = generator.flow_from_directory(
            "dataset/imagenet/train",
            target_size=(224, 224),
            batch_size=64,
        )
        test_datagen = ImageDataGenerator(rescale=1. / 255)  # 验证集不用增强
        validation_generator = test_datagen.flow_from_directory(
            'dataset/imagenet/val',
            target_size=(224, 224),
            batch_size=64
        )
        # Load model
        weights_file = "weights/imagenet_model.h5"
        if os.path.exists(weights_file) and is_train == False:
            self.model.load_weights(weights_file, by_name=True)
            print("Model loaded.")

        lr_reducer = ReduceLROnPlateau(monitor='val_acc', factor=np.sqrt(0.1),
                                       cooldown=0, patience=5, min_lr=1e-5)
        model_checkpoint = ModelCheckpoint(weights_file, monitor="val_acc", save_best_only=True,
                                           save_weights_only=True, verbose=1)
        callbacks = [lr_reducer, model_checkpoint]

        image_list = list()
        with open('prepare_data.txt', 'r') as file:
            lines = file.readlines()
            print(len(lines))
            for line in lines:
                temp = line.split(',,')
                if (temp[0][17] == 't'):
                    image_list.append([temp[0], int(temp[1])])
        random.shuffle(image_list)
        print("#############")
        if (is_train == True):
            self.model.fit_generator(train_generator,
                                     steps_per_epoch=len(image_list) // batch_size, epochs=200,
                                     callbacks=callbacks,
                                     validation_data=validation_generator,
                                     validation_steps=x_test_temp.shape[0] // batch_size, verbose=1)
