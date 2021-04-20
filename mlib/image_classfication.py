import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Check https://keras.io/applications/ or https://www.tensorflow.org/api_docs/python/tf/keras/applications
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
model = MobileNetV2(weights='imagenet')

print('MobileNetV2 Model loaded.')

# Model saved with Keras model.save()
# MODEL_PATH = 'models/model_p1/model.h5'
# model = load_model(MODEL_PATH, compile=False)
print('MobileNetV2 Model loaded. Start serving...')

def model_predict(img, model):
    img = img.resize((224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='tf')

    preds = model.predict(x)
    return preds
