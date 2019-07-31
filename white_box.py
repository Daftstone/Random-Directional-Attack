from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import time

sys.path.append("cleverhans")

import tensorflow as tf
from tensorflow.python.platform import flags
import numpy as np

from cleverhans_tutorials import check_installation
from cleverhans.utils import AccuracyReport
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.utils_tf import model_eval
from utils import get_value, get_model
from datasets import data_fmnist, data_cifar10, data_mnist, data_svhn, data_imagenet
from attacks import get_adv_examples
from keras import backend as K
from utils import RDA
from utils import GetGradient

# Create TF session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

FLAGS = flags.FLAGS

NB_EPOCHS = 100
BATCH_SIZE = 128


def attack(nb_epochs=NB_EPOCHS, batch_size=BATCH_SIZE,
           ):

    # keras.layers.core.K.set_learning_phase(0)

    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # get data
    dataset_dir = {"mnist": data_mnist, "cifar10": data_cifar10, "fmnist": data_fmnist, 'svhn': data_svhn,
                   'imagenet': data_imagenet}
    x_train, y_train, x_test, y_test = dataset_dir[FLAGS.data]()

    # Obtain Image Parameters
    img_rows, img_cols, nchannels = x_train.shape[1:4]
    nb_classes = y_train.shape[1]

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
                                          nchannels))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))

    # Define TF model graph
    Model = get_model(FLAGS.data, 10)
    Model.train(x_train, y_train, x_test, y_test, batch_size, nb_epochs, FLAGS.is_train)
    model = Model.model
    eval_par = {'batch_size': batch_size}
    acc = model_eval(sess, x, y, model(x), x_test, y_test, args=eval_par)
    print('Test accuracy on legitimate examples: %.4f' % acc)

    # select successful classified examples
    pred_all = model.predict(x_test)
    classfied_flag = np.argmax(pred_all, axis=-1) == np.argmax(y_test, axis=-1)
    x_test = x_test[classfied_flag]
    y_test = y_test[classfied_flag]
    acc = model_eval(sess, x, y, model(x), x_test, y_test, args=eval_par)
    print('Test accuracy on successfully classified examples(should be 1.00): %.4f' % acc)

    wrap = KerasModelWrapper(model)
    x_test = x_test
    y_test = y_test

    # Initialize the Fast Gradient Sign Method (FGSM) attack object and graph
    grad_temp = GetGradient(wrap, sess=sess)
    params = {'clip_min': 0., 'clip_max': 1.}
    grad = grad_temp.generate(x, **params)
    # Consider the attack to be constant
    gradient_tensor = tf.stop_gradient(grad)

    # Evaluate the accuracy of the trained model on adversarial examples
    gradient = get_value(sess, x, gradient_tensor, x_test, batch_size=batch_size)
    print('Test accuracy on fgsm examples: %0.4f\n' % model_eval(sess, x, y, model(x),
                                                                 np.clip(
                                                                     x_test + np.sign(gradient) * FLAGS.eps, 0,
                                                                     1), y_test, args=eval_par))

    d1 = get_adv_examples(sess, wrap, "l.l.class", x_test, y_test)
    print('Test accuracy on lfgsm examples: %0.4f\n' % model_eval(sess, x, y, model(x), d1, y_test,
                                                                  args=eval_par))
    d2 = get_adv_examples(sess, wrap, "bim", x_test, y_test)
    print('Test accuracy on bim examples: %0.4f\n' % model_eval(sess, x, y, model(x), d2, y_test,
                                                                args=eval_par))
    d3 = get_adv_examples(sess, wrap, "mi-fgsm", x_test, y_test)
    print('Test accuracy on mi-fgsm examples: %0.4f\n' % model_eval(sess, x, y, model(x), d3, y_test,
                                                                    args=eval_par))
    
    # Evaluate the accuracy of the trained model on RDA examples
    rda_direction = RDA(model, x_test, y_test, gradient, FLAGS.eps, batch_size, FLAGS.max_angle, FLAGS.nb_dimensions)
    x_adv = np.clip(x_test + np.sign(rda_direction) * FLAGS.eps, 0, 1)
    acc = model_eval(sess, x, y, model(x), x_adv, y_test, args=eval_par)
    print('Test accuracy on RDA examples: %0.4f\n' % acc)

    report.clean_train_adv_eval = acc
    return report


def main(argv=None):
    check_installation(__file__)

    attack(nb_epochs=FLAGS.nb_epochs,
           batch_size=FLAGS.batch_size)


if __name__ == '__main__':
    flags.DEFINE_integer('nb_epochs', NB_EPOCHS,
                         'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', BATCH_SIZE, 'Size of training batches')
    flags.DEFINE_integer('max_angle', 180, 'Maximum angle of rotation')
    flags.DEFINE_integer('nb_dimensions', 10, 'Number of dimensions selected')
    flags.DEFINE_string('data', "mnist", 'data name')
    flags.DEFINE_boolean('is_train', False,
                         'train online or load from file')
    flags.DEFINE_float('eps', 0.1,
                       'eps')
    tf.app.run()
