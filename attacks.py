import tensorflow as tf
import numpy as np
from keras import backend as K
from utils import get_value, get_adv
from tensorflow.python.platform import flags

from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import BasicIterativeMethod
from cleverhans.attacks import MomentumIterativeMethod

FLAGS = flags.FLAGS


def get_adv_examples(sess, wrap, attack_type, X, Y):
    """
        detect adversarial examples
        :param sess: target model session
        :param wrap: wrap model
        :param attack_type:  attack for generating adversarial examples
        :param X: examples to be attacked
        :param Y: correct label of the examples
        :return: x_adv: adversarial examples
    """
    x = tf.placeholder(tf.float32, shape=(None, X.shape[1], X.shape[2],
                                          X.shape[3]))
    y = tf.placeholder(tf.float32, shape=(None, Y.shape[1]))
    adv_label = np.copy(Y)
    batch_size = 128

    # Define attack method parameters
    if (attack_type == 'fgsm'):
        attack_params = {
            'eps': FLAGS.eps,
            'clip_min': 0.,
            'clip_max': 1.
        }
        attack_object = FastGradientMethod(wrap, sess=sess)
    elif (attack_type == 'mi-fgsm'):
        attack_object = MomentumIterativeMethod(wrap, back='tf', sess=sess)
        if (FLAGS.eps < 0.05):
            attack_params = {'eps': FLAGS.eps, 'eps_iter': FLAGS.eps,
                             'nb_iter': 10, 'clip_min': 0.,
                             'clip_max': 1., 'batch_size': FLAGS.batch_size
                             }
        else:
            attack_params = {'eps': FLAGS.eps, 'eps_iter': 0.05,
                             'nb_iter': 10, 'clip_min': 0.,
                             'clip_max': 1., 'batch_size': FLAGS.batch_size
                             }
    elif (attack_type == 'bim'):
        attack_object = BasicIterativeMethod(wrap, back='tf', sess=sess)
        if (FLAGS.eps < 0.05):
            attack_params = {'eps': FLAGS.eps, 'eps_iter': FLAGS.eps,
                             'nb_iter': 10, 'clip_min': 0.,
                             'clip_max': 1., 'batch_size': FLAGS.batch_size
                             }
        else:
            attack_params = {'eps': FLAGS.eps, 'eps_iter': 0.05,
                             'nb_iter': 10, 'clip_min': 0.,
                             'clip_max': 1., 'batch_size': FLAGS.batch_size
                             }
    elif (attack_type == 'l.l.class'):
        attack_object = BasicIterativeMethod(wrap, back='tf', sess=sess)
        if (FLAGS.eps < 0.05):
            attack_params = {'eps': FLAGS.eps, 'eps_iter': FLAGS.eps,
                             'nb_iter': 10, 'clip_min': 0.,
                             'clip_max': 1., 'y_target': y, 'batch_size': FLAGS.batch_size
                             }
        else:
            attack_params = {'eps': FLAGS.eps, 'eps_iter': 0.05,
                             'nb_iter': 10, 'clip_min': 0.,
                             'clip_max': 1., 'y_target': y, 'batch_size': FLAGS.batch_size
                             }
        ll = np.argmin(Y, axis=-1)
        for i in range(len(Y)):
            ind = ll[i]
            adv_label[i] = np.zeros([10])
            adv_label[i, ind] = 1
    adv_x = attack_object.generate(x, **attack_params)

    # Get adversarial examples
    if (attack_type == 'l.l.class'):
        x_adv = get_adv(sess, x, y, adv_x, X, adv_label, batch_size=FLAGS.batch_size)
    else:
        x_adv = get_adv(sess, x, y, adv_x, X, Y, batch_size=FLAGS.batch_size)
    return x_adv
