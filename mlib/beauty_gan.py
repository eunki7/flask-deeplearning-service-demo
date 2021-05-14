# Refer : https://github.com/Honlan/BeautyGAN
# Tensorflow 1.x version compatibility
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Tensorflow 2.x
# import tensorflow as tf

# import packages
import numpy as np
import os
import glob
from imageio import imread, imsave
import cv2

# Model Paths
MODEL_GAN_PATH = os.path.join('models', 'model_p2', 'model.meta')
MODEL_GAN_CHECKPOINT_PATH = os.path.join('models', 'model_p2')
MODEL_FACE_LANDMARK_PATH = os.path.join('models', 'model_dlib', 'shape_predictor_5_face_landmarks.dat')

def preprocess(img):
    return (img / 255. - 0.5) * 2

def deprocess(img):
    return (img + 1) / 2

def postprocess(img):
    return (img  * 255).astype(np.uint8)

def predict_single_or_all(ori_img, mp_img=None):
    """Function for nomakeup image to all makeup images methods."""
    
    # image size
    img_size = 256
    
    # loads makeup and no_makeup images
    no_makeup = cv2.resize(np.asarray(ori_img), (img_size, img_size))
    
    # loads local test image
    # no_makeup = cv2.resize(imread(os.path.join('mlib','imgs', 'no_makeup', 'xfsy_0071.png')), (img_size, img_size))

    X_img = np.expand_dims(preprocess(no_makeup), 0)
    makeups = glob.glob(os.path.join('mlib', 'imgs', 'makeup', '*.*'))
    
    # use numpy and result image size set
    if mp_img is None :
        result = np.ones((2 * img_size, (len(makeups) + 1) * img_size, 3))
        result[img_size: 2 *  img_size, :img_size] = no_makeup / 255.
    else :
        result = np.ones((img_size, 2 * img_size, 3))
        print(result.shape)
        result[:img_size, :img_size] = no_makeup / 255.
        imsave('result_all.jpg', result)
    
    # initialize tensorflow
    tf.reset_default_graph()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # model & checkpoint loads
    saver = tf.train.import_meta_graph(MODEL_GAN_PATH)
    saver.restore(sess, tf.train.latest_checkpoint(MODEL_GAN_CHECKPOINT_PATH))
    graph = tf.get_default_graph()
    
    # get tensor by name
    X = graph.get_tensor_by_name('X:0')
    Y = graph.get_tensor_by_name('Y:0')
    Xs = graph.get_tensor_by_name('generator/xs:0')

    if mp_img is None :
        # generate all makeup image
        for i in range(len(makeups)):
            makeup = cv2.resize(imread(makeups[i]), (img_size, img_size))
            Y_img = np.expand_dims(preprocess(makeup), 0)
            
            # run beauty gan
            Xs_ = sess.run(Xs, feed_dict={X: X_img, Y: Y_img})
            Xs_ = deprocess(Xs_)
            
            # image calculate
            result[:img_size, (i + 1) * img_size: (i + 2) * img_size] = makeup / 255.
            result[img_size: 2 * img_size, (i + 1) * img_size: (i + 2) * img_size] = Xs_[0]
            
            # test image save
            # imsave('result_single.jpg', result)
    else :
            Y_img = np.expand_dims(preprocess(cv2.resize(np.asarray(mp_img), (img_size, img_size))), 0)
            
            # run beauty gan
            Xs_ = sess.run(Xs, feed_dict={X: X_img, Y: Y_img})
            Xs_ = deprocess(Xs_)
            
            # result[:img_size, 1 * img_size: 2 * img_size] = Y_img / 255.
            result[:img_size, img_size: 2 * img_size] = Xs_[0]
            
            # test image save
            # imsave('result_all.jpg', result)

    return postprocess(result)