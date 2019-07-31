import numpy as np
import os
import random
import math
import warnings
import tensorflow as tf
from tensorflow.python.platform import flags
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense, Activation, Flatten, add, Input
from keras.layers import MaxPooling2D, Conv2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
import keras
from keras.models import Model
from keras.applications.mobilenet_v2 import MobileNetV2

from models.mnist_model import MNIST_model
from models.fmnist_model import FMNIST_model
from models.svhn_model import SVHN_model
from models.cifar10_model import CIFAR10_model
from models.imagenet_model import IMAGENET_model
from cleverhans import utils_tf
from cleverhans.compat import reduce_sum, reduce_mean
from cleverhans.compat import reduce_max
from cleverhans.compat import softmax_cross_entropy_with_logits
from cleverhans.attacks import Attack


FLAGS = flags.FLAGS

class GetGradient(Attack):
    """
    This attack was originally implemented by Goodfellow et al. (2015) with the
    infinity norm (and is known as the "Fast Gradient Sign Method"). This
    implementation extends the attack to other norms, and is therefore called
    the Fast Gradient Method.
    Paper link: https://arxiv.org/abs/1412.6572

    :param model: cleverhans.model.Model
    :param sess: optional tf.Session
    :param dtypestr: dtype of the data
    :param kwargs: passed through to super constructor
    """

    def __init__(self, model, sess=None, dtypestr='float32', **kwargs):
        """
        Create a FastGradientMethod instance.
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """

        super(GetGradient, self).__init__(model, sess, dtypestr, **kwargs)
        self.feedable_kwargs = ('eps', 'y', 'y_target', 'clip_min', 'clip_max')
        self.structural_kwargs = ['ord', 'sanity_checks']

    def generate(self, x, **kwargs):
        """
        Returns the graph for Fast Gradient Method adversarial examples.

        :param x: The model's symbolic inputs.
        :param kwargs: See `parse_params`
        """
        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        labels, _nb_classes = self.get_or_guess_labels(x, kwargs)

        return get_gradient(
            x,
            self.model.get_logits(x),
            y=labels,
            eps=self.eps,
            ord=self.ord,
            clip_min=self.clip_min,
            clip_max=self.clip_max,
            targeted=(self.y_target is not None),
            sanity_checks=self.sanity_checks)

    def parse_params(self,
                     eps=0.3,
                     ord=np.inf,
                     y=None,
                     y_target=None,
                     clip_min=None,
                     clip_max=None,
                     sanity_checks=True,
                     **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.

        Attack-specific parameters:

        :param eps: (optional float) attack step size (input variation)
        :param ord: (optional) Order of the norm (mimics NumPy).
                    Possible values: np.inf, 1 or 2.
        :param y: (optional) A tensor with the true labels. Only provide
                  this parameter if you'd like to use true labels when crafting
                  adversarial samples. Otherwise, model predictions are used as
                  labels to avoid the "label leaking" effect (explained in this
                  paper: https://arxiv.org/abs/1611.01236). Default is None.
                  Labels should be one-hot-encoded.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        :param sanity_checks: bool, if True, include asserts
          (Turn them off to use less runtime / memory or for unit tests that
          intentionally pass strange input)
        """
        # Save attack-specific parameters

        self.eps = eps
        self.ord = ord
        self.y = y
        self.y_target = y_target
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.sanity_checks = sanity_checks

        if self.y is not None and self.y_target is not None:
            raise ValueError("Must not set both y and y_target")
        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, int(1), int(2)]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")

        if len(kwargs.keys()) > 0:
            warnings.warn("kwargs is unused and will be removed on or after "
                          "2019-04-26.")

        return True
def get_gradient(x,
         logits,
         y=None,
         eps=0.3,
         ord=np.inf,
         clip_min=None,
         clip_max=None,
         targeted=False,
         sanity_checks=True):
    """
    TensorFlow implementation of the Fast Gradient Method.
    :param x: the input placeholder
    :param logits: output of model.get_logits
    :param y: (optional) A placeholder for the true labels. If targeted
              is true, then provide the target label. Otherwise, only provide
              this parameter if you'd like to use true labels when crafting
              adversarial samples. Otherwise, model predictions are used as
              labels to avoid the "label leaking" effect (explained in this
              paper: https://arxiv.org/abs/1611.01236). Default is None.
              Labels should be one-hot-encoded.
    :param eps: the epsilon (input variation parameter)
    :param ord: (optional) Order of the norm (mimics NumPy).
                Possible values: np.inf, 1 or 2.
    :param clip_min: Minimum float value for adversarial example components
    :param clip_max: Maximum float value for adversarial example components
    :param targeted: Is the attack targeted or untargeted? Untargeted, the
                     default, will try to make the label incorrect. Targeted
                     will instead try to move in the direction of being more
                     like y.
    :return: a tensor for the adversarial example
    """

    asserts = []

    # If a data range was specified, check that the input was in that range
    if clip_min is not None:
        asserts.append(utils_tf.assert_greater_equal(
            x, tf.cast(clip_min, x.dtype)))

    if clip_max is not None:
        asserts.append(utils_tf.assert_less_equal(x, tf.cast(clip_max, x.dtype)))

    # Make sure the caller has not passed probs by accident
    assert logits.op.type != 'Softmax'

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = reduce_max(logits, 1, keepdims=True)
        y = tf.to_float(tf.equal(logits, preds_max))
        y = tf.stop_gradient(y)
    y = y / reduce_sum(y, 1, keepdims=True)

    # Compute loss
    loss = softmax_cross_entropy_with_logits(labels=y, logits=logits)
    if targeted:
        loss = -loss

    # Define gradient of loss wrt input
    grad, = tf.gradients(loss, x)
    optimal_perturbation = grad
    # optimal_perturbation = optimize_linear(grad, eps, ord)
    # Add perturbation to original example to obtain adversarial example
    adv_x = optimal_perturbation

    if sanity_checks:
        with tf.control_dependencies(asserts):
            adv_x = tf.identity(adv_x)

    return adv_x

def get_model(dataname, nb_classes):
    if (dataname == "mnist"):
        model = MNIST_model(nb_classes=nb_classes)
    elif (dataname == 'cifar10'):
        model = CIFAR10_model(nb_classes=nb_classes)
    elif (dataname == 'fmnist'):
        model = FMNIST_model(nb_classes=nb_classes)
    elif (dataname == 'svhn'):
        model = SVHN_model(nb_classes=nb_classes)
    elif (dataname == 'imagenet'):
        model = IMAGENET_model(nb_classes=nb_classes)
    else:
        model = None
        print("unknown model!!!")
    return model

def get_value(sess, x, distortion, X_test=None,
                   feed=None, batch_size=128):
    assert batch_size, "Batch size was not given in args dict"
    if X_test is None:
        raise ValueError("X_test argument "
                         "must be supplied.")

    # Init result var
    result = np.ndarray(X_test.shape)

    with sess.as_default():
        # Compute number of batches
        nb_batches = int(math.ceil(float(len(X_test)) / batch_size))
        assert nb_batches * batch_size >= len(X_test)

        X_cur = np.zeros((batch_size,) + X_test.shape[1:],
                         dtype=X_test.dtype)
        for batch in range(nb_batches):

            # Must not use the `batch_indices` function here, because it
            # repeats some examples.
            # It's acceptable to repeat during training, but not eval.
            start = batch * batch_size
            end = min(len(X_test), start + batch_size)

            # The last batch may be smaller than all others. This should not
            # affect the accuarcy disproportionately.
            cur_batch_size = end - start
            X_cur[:cur_batch_size] = X_test[start:end]
            feed_dict = {x: X_cur}
            if feed is not None:
                feed_dict.update(feed)
            cur_preds = distortion.eval(feed_dict=feed_dict)
            result[start:end] = cur_preds[:cur_batch_size]
        assert end >= len(X_test)

    return result

def get_adv(sess, x, y, adv, X_test=None, Y_test=None,
            feed=None, batch_size=128):
    """
    Compute the accuracy of a TF model on some data
    :param sess: TF session to use when training the graph
    :param x: input placeholder
    :param y: output placeholder (for labels)
    :param predictions: model output predictions
    :param X_test: numpy array with training inputs
    :param Y_test: numpy array with training outputs
    :param feed: An optional dictionary that is appended to the feeding
             dictionary before the session runs. Can be used to feed
             the learning phase of a Keras model for instance.
    :param args: dict or argparse `Namespace` object.
                 Should contain `batch_size`
    :return: a float with the accuracy value
    """

    assert batch_size, "Batch size was not given in args dict"
    if X_test is None or Y_test is None:
        raise ValueError("X_test argument and Y_test argument "
                         "must be supplied.")

    adv_x = np.ndarray(X_test.shape, dtype=X_test.dtype)

    with sess.as_default():
        # Compute number of batches
        nb_batches = int(math.ceil(float(len(X_test)) / batch_size))
        assert nb_batches * batch_size >= len(X_test)

        X_cur = np.zeros((batch_size,) + X_test.shape[1:],
                         dtype=X_test.dtype)
        Y_cur = np.zeros((batch_size,) + Y_test.shape[1:],
                         dtype=Y_test.dtype)
        for batch in range(nb_batches):

            # Must not use the `batch_indices` function here, because it
            # repeats some examples.
            # It's acceptable to repeat during training, but not eval.
            start = batch * batch_size
            end = min(len(X_test), start + batch_size)

            # The last batch may be smaller than all others. This should not
            # affect the accuarcy disproportionately.
            cur_batch_size = end - start
            X_cur[:cur_batch_size] = X_test[start:end]
            Y_cur[:cur_batch_size] = Y_test[start:end]
            # feed_dict = {x: X_cur, y: Y_cur}
            feed_dict = {x: X_cur, y: Y_cur}
            if feed is not None:
                feed_dict.update(feed)
            adv_x[start:end] = sess.run(adv, feed_dict=feed_dict)[:cur_batch_size]
        assert end >= len(X_test)

    return adv_x

def get_metrix_index(per_number, select_dimension, all_dimension):
    metrix_index = np.ndarray([per_number * 2, select_dimension // 2, 6])
    index = np.array([i for i in range(0, all_dimension)])
    np.random.shuffle(index)
    count = 0
    for j in range(-per_number, per_number + 1):
        if (j == 0):
            continue
        np.random.shuffle(index)
        select_index = index[:select_dimension]
        for k in range(select_dimension // 2):
            ang = j
            metrix_index[count, k, 0] = select_index[2 * k]
            metrix_index[count, k, 1] = select_index[2 * k + 1]
            metrix_index[count, k, 2] = np.cos(ang * np.pi / 180.)
            metrix_index[count, k, 3] = -np.sin(ang * np.pi / 180.)
            metrix_index[count, k, 4] = np.sin(ang * np.pi / 180.)
            metrix_index[count, k, 5] = np.cos(ang * np.pi / 180.)
        count += 1
    return metrix_index

def cal_dot(metrix, info):
    # print(metrix.shape, info.shape)
    tt = np.copy(metrix)
    for j in range(len(metrix)):
        for i in range(len(info)):
            temp1 = info[i, 2] * metrix[j, int(info[i, 0]), 0] + info[i, 3] * metrix[j, int(info[i, 1]), 0]
            temp2 = info[i, 4] * metrix[j, int(info[i, 0]), 0] + info[i, 5] * metrix[j, int(info[i, 1]), 0]
            tt[j, int(info[i, 0]), 0] = temp1
            tt[j, int(info[i, 1]), 0] = temp2
    return tt


def RDA(model, x_test, y_test, direction_total, eps, batch_size=128, angle=180, nb_dimension=10):
    total_dimensions = x_test.shape[1] * x_test.shape[2] * x_test.shape[3]
    direction = np.copy(direction_total)
    direction = np.reshape(direction, [len(direction), -1, 1])
    end = 0
    true_class = np.argmax(y_test, axis=1)
    init_predict = model.predict(np.clip(x_test + np.sign(np.reshape(direction_total, x_test.shape)) * eps, 0, 1))
    cur_best = np.diag(init_predict[:, true_class])

    direction_batch = list()
    true_class_batch = list()
    cur_best_batch = list()
    success_batch = list()
    x_test_batch = list()
    index_batch = list()
    jump_batch = list()

    count = 0
    while (count != len(x_test)):
        cur_len = len(direction_batch)
        for i in range(batch_size - cur_len):
            if (end == len(x_test)):
                break
            direction_batch.append(direction[end])
            true_class_batch.append(true_class[end])
            cur_best_batch.append(cur_best[end])
            success_batch.append(False)
            x_test_batch.append(x_test[end])
            index_batch.append(end)
            jump_batch.append(False)
            end += 1
        rot_matrix = get_metrix_index(angle, nb_dimension, total_dimensions)

        direction_batch_temp = direction_batch.copy()
        true_class_batch_temp = true_class_batch.copy()
        cur_best_batch_temp = cur_best_batch.copy()
        success_batch_temp = success_batch.copy()
        x_test_batch_temp = x_test_batch.copy()
        jump_batch_temp = jump_batch.copy()
        index_batch_temp = [i for i in range(len(index_batch))]

        random_list = np.arange(2 * angle)
        np.random.shuffle(random_list)
        for j in range(2 * angle):
            if (len(direction_batch_temp) == 0):
                break
            direction_batch_temp_temp = cal_dot(np.array(direction_batch_temp), rot_matrix[random_list[j]])
            local_p = model.predict(np.clip(np.array(x_test_batch_temp) + np.sign(
                np.reshape(direction_batch_temp_temp, (len(direction_batch_temp_temp),) + x_test.shape[1:])) * eps, 0,
                                            1))
            test1 = np.argmax(local_p, axis=-1) != true_class_batch_temp
            for i in range(len(direction_batch_temp)):
                if (test1[i] == True):
                    success_batch_temp[i] = True
                    direction_batch_temp[i] = direction_batch_temp_temp[i]
                    jump_batch_temp[i] = True
                    cur_best_batch_temp[i] = local_p[i, true_class_batch_temp[i]]
                elif (local_p[i, true_class_batch_temp[i]] < cur_best_batch_temp[i]):
                    direction_batch_temp[i] = direction_batch_temp_temp[i]
                    jump_batch_temp[i] = True
                    cur_best_batch_temp[i] = local_p[i, true_class_batch_temp[i]]
            lens = len(direction_batch_temp)
            for i in range(lens - 1, -1, -1):
                if (jump_batch_temp[i] == True):
                    direction_batch[index_batch_temp[i]] = direction_batch_temp[i]
                    success_batch[index_batch_temp[i]] = success_batch_temp[i]
                    cur_best_batch[index_batch_temp[i]] = cur_best_batch_temp[i]
                    jump_batch[index_batch_temp[i]] = jump_batch_temp[i]

                    direction_batch_temp.pop(i)
                    true_class_batch_temp.pop(i)
                    cur_best_batch_temp.pop(i)
                    success_batch_temp.pop(i)
                    x_test_batch_temp.pop(i)
                    index_batch_temp.pop(i)
                    jump_batch_temp.pop(i)
        lens = len(direction_batch)

        for i in range(lens - 1, -1, -1):
            if (success_batch[i] == True or jump_batch[i] == False):
                direction[index_batch[i]] = direction_batch[i]
                count += 1

                direction_batch.pop(i)
                true_class_batch.pop(i)
                cur_best_batch.pop(i)
                success_batch.pop(i)
                x_test_batch.pop(i)
                index_batch.pop(i)
                jump_batch.pop(i)
            else:
                jump_batch[i] = False
    return np.reshape(direction, x_test.shape)

def mnist_model(nb_filters=64, input_shape=(None, 28, 28, 1)):
    model = keras.Sequential()
    for scale in range(3):
        if (scale == 0):
            model.add(keras.layers.Convolution2D(filters=nb_filters << scale, kernel_size=(3, 3),
                                                 kernel_initializer=keras.initializers.he_normal(),
                                                 activation='relu',
                                                 padding='same', input_shape=input_shape[1:]))
        else:
            model.add(keras.layers.Convolution2D(filters=nb_filters << scale, kernel_size=(3, 3),
                                                 kernel_initializer=keras.initializers.he_normal(),
                                                 activation='relu',
                                                 padding='same'))
        model.add(keras.layers.Convolution2D(filters=nb_filters << (scale + 1), kernel_size=(3, 3),
                                             kernel_initializer=keras.initializers.he_normal(), activation='relu',
                                             padding='same'))
        model.add(keras.layers.AveragePooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.Dense(10))
    model.add(keras.layers.Activation("softmax"))
    return model


def svhn_model(nb_filters=64, input_shape=(None, 32, 32, 3)):
    model = keras.Sequential()

    model.add(Conv2D(nb_filters, (3, 3), input_shape=input_shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(nb_filters, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(nb_filters * 2, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(nb_filters * 2, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    return model


def cifar10_model(nb_filters=64, input_shape=(None, 32, 32, 3)):
    def resnet(input):
        def residual_block(intput, out_channel, increase=False):
            if increase:
                stride = (2, 2)
            else:
                stride = (1, 1)

            pre_bn = BatchNormalization()(intput)
            pre_relu = Activation('relu')(pre_bn)

            conv_1 = Conv2D(out_channel, kernel_size=(3, 3), strides=stride, padding='same',
                            kernel_initializer="he_normal",
                            kernel_regularizer=keras.regularizers.l2(0.0001))(pre_relu)
            bn_1 = BatchNormalization()(conv_1)
            relu1 = Activation('relu')(bn_1)
            conv_2 = Conv2D(out_channel, kernel_size=(3, 3), strides=(1, 1), padding='same',
                            kernel_initializer="he_normal",
                            kernel_regularizer=keras.regularizers.l2(0.0001))(relu1)
            if increase:
                projection = Conv2D(out_channel,
                                    kernel_size=(1, 1),
                                    strides=(2, 2),
                                    padding='same',
                                    kernel_initializer="he_normal",
                                    kernel_regularizer=keras.regularizers.l2(0.0001))(intput)
                block = add([conv_2, projection])
            else:
                block = add([intput, conv_2])
            return block

        # build model
        # total layers = stack_n * 3 * 2 + 2
        # stack_n = 5 by default, total layers = 32
        # input: 32x32x3 output: 32x32x16
        x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
                   kernel_initializer="he_normal",
                   kernel_regularizer=keras.regularizers.l2(0.0001))(input)

        # input: 32x32x16 output: 32x32x16
        for _ in range(5):
            x = residual_block(x, 16, False)

        # input: 32x32x16 output: 16x16x32
        x = residual_block(x, 32, True)
        for _ in range(1, 5):
            x = residual_block(x, 32, False)

        # input: 16x16x32 output: 8x8x64
        x = residual_block(x, 64, True)
        for _ in range(1, 5):
            x = residual_block(x, 64, False)

        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = GlobalAveragePooling2D()(x)

        # input: 64 output: 10
        x = Dense(10, activation='softmax',
                  kernel_initializer="he_normal",
                  kernel_regularizer=keras.regularizers.l2(0.0001))(x)
        return x

    img_input = Input(shape=(32, 32, 3))
    output = resnet(img_input)
    resnet = Model(img_input, output)
    return resnet


def imagenet_model(nb_filters=64, input_shape=(None, 224, 224, 3)):
    model = MobileNetV2(weights=None, classes=10)

    return model


def train(model, x_train, y_train, x_test, y_test, batch_size=128, nb_epochs=60, is_train=True):
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
    optimizer = Adam(lr=1e-3)  # Using Adam instead of SGD to speed up training
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    generator = ImageDataGenerator(rotation_range=15,
                                   width_shift_range=5. / 32,
                                   height_shift_range=5. / 32,
                                   horizontal_flip=True)
    generator.fit(x_train, seed=0)
    # Load model
    weights_file = "weights/%s_model_temp.h5" % FLAGS.data
    if os.path.exists(weights_file) and is_train == False:
        model.load_weights(weights_file)
        print("Model loaded.")
        print("#############")

    lr_reducer = ReduceLROnPlateau(monitor='val_acc', factor=np.sqrt(0.1),
                                   cooldown=0, patience=5, min_lr=1e-5)
    model_checkpoint = ModelCheckpoint(weights_file, monitor="val_acc", save_best_only=True,
                                       save_weights_only=True, verbose=1)

    callbacks = [lr_reducer, model_checkpoint]
    if (is_train == True):
        print("begin train.")
        model.fit_generator(generator.flow(x_train, y_train, batch_size=batch_size),

                            steps_per_epoch=len(x_train) // batch_size, epochs=nb_epochs,
                            callbacks=callbacks,
                            validation_data=(x_test, y_test),
                            validation_steps=x_test.shape[0] // batch_size, verbose=1)
    return model


def imagenet_train(model, x_train, y_train, x_test, y_test, batch_size=64, nb_epochs=250, is_train=True):
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
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
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
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_generator = test_datagen.flow_from_directory(
        'dataset/imagenet/val',
        target_size=(224, 224),
        batch_size=64
    )
    # Load model
    weights_file = "weights/imagenet_model_temp.h5"
    if os.path.exists(weights_file) and is_train == False:
        model.load_weights(weights_file, by_name=True)
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
        model.fit_generator(train_generator,
                            steps_per_epoch=len(image_list) // batch_size, epochs=200,
                            callbacks=callbacks,
                            validation_data=validation_generator,
                            validation_steps=x_test_temp.shape[0] // batch_size, verbose=1)
    return model
