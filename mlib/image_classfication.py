# Refer : https://github.com/mtobeiyf/keras-flask-deploy-webapp

# 2.x
# import tensorflow as tf

# 1.x
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow import keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Model check
# https://keras.io/applications/ 
# https://www.tensorflow.org/api_docs/python/tf/keras/applications
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
model = MobileNetV2(weights='imagenet')

# Model save
# MODEL_PATH = 'models/model_p1/model.h5'
# model.save(MODEL_PATH)

# Model load
# model = load_model(MODEL_PATH, compile=False)

# Check
# print('MobileNetV2 Model load successed.')

# Image classfication 
def model_predict(img, model):
    # Image resizing
    img = img.resize((224, 224))

    # Image to array
    x = image.img_to_array(img)

    # Check
    # x = np.true_divide(x, 255)

    # Axis convert
    x = np.expand_dims(x, axis=0)

    # Image preprocessing
    x = preprocess_input(x, mode='tf')

    # predict
    preds = model.predict(x)

    # return value
    return preds
