# 참조 : https://github.com/Honlan/BeautyGAN
# 텐서플로우 관련 패키지
import tensorflow.compat.v1 as tf

# 텐서플로우 하위 호환
tf.disable_v2_behavior()
# import tensorflow as tf

import numpy as np
import os
import glob
from imageio import imread, imsave
import cv2

MODEL_PATH = os.path.join('models', 'model_p2', 'model.meta')
MODEL_CHECKPOINT_PATH = os.path.join('models', 'model_p2')

def preprocess(img):
    return (img / 255. - 0.5) * 2

def deprocess(img):
    return (img + 1) / 2

def model_predict():
    batch_size = 1
    img_size = 256
    no_makeup = cv2.resize(imread(os.path.join('mlib','imgs', 'no_makeup', 'xfsy_0069.jpg')), (img_size, img_size))
    X_img = np.expand_dims(preprocess(no_makeup), 0)
    makeups = glob.glob(os.path.join('mlib', 'imgs', 'makeup', '*.*'))
    result = np.ones((2 * img_size, (len(makeups) + 1) * img_size, 3))
    result[img_size: 2 *  img_size, :img_size] = no_makeup / 255.

    tf.reset_default_graph()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.import_meta_graph(MODEL_PATH)
    saver.restore(sess, tf.train.latest_checkpoint(MODEL_CHECKPOINT_PATH))

    graph = tf.get_default_graph()
    X = graph.get_tensor_by_name('X:0')
    Y = graph.get_tensor_by_name('Y:0')
    Xs = graph.get_tensor_by_name('generator/xs:0')

    for i in range(len(makeups)):
        makeup = cv2.resize(imread(makeups[i]), (img_size, img_size))
        Y_img = np.expand_dims(preprocess(makeup), 0)
        Xs_ = sess.run(Xs, feed_dict={X: X_img, Y: Y_img})
        Xs_ = deprocess(Xs_)
        result[:img_size, (i + 1) * img_size: (i + 2) * img_size] = makeup / 255.
        result[img_size: 2 * img_size, (i + 1) * img_size: (i + 2) * img_size] = Xs_[0]

        imsave('result.jpg', result)
    return result
