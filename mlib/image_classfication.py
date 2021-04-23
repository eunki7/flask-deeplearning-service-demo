# 참조 : https://github.com/mtobeiyf/keras-flask-deploy-webapp
# 텐서플로우 관련 패키지
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# 모델 확인 
# https://keras.io/applications/ 
# https://www.tensorflow.org/api_docs/python/tf/keras/applications
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
model = MobileNetV2(weights='imagenet')

# 모델 저장
# MODEL_PATH = 'models/model_p1/model.h5'
# model.save()

# 모델 로드
# model = load_model(MODEL_PATH, compile=False)

# 확인
# print('MobileNetV2 모델 로드 완료.')

# 이미지 분류 예측 메소드
def model_predict(img, model):
    # 리사이징(조절 가능)
    img = img.resize((224, 224))

    # 이미지 배열 변환
    x = image.img_to_array(img)

    # 필요시 주석 해제
    # x = np.true_divide(x, 255)

    # 축 변경
    x = np.expand_dims(x, axis=0)

    # 이미지 전처리
    x = preprocess_input(x, mode='tf')

    # 예측
    preds = model.predict(x)

    # 반환
    return preds
