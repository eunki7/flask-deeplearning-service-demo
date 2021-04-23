import os
import sys

# 플라스크 관련
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect, make_response
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# 딥러닝 기반 패키지
from mlib import image_classfication as img_cf
from mlib import beauty_gan as img_bt

# 유틸리티
import numpy as np
from utils.util import base64_to_pil

# 플라스크 선언
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index_ex1():
    # 인덱스(이미지 분류)
    return render_template('index_ex1.html')

@app.route('/beauty', methods=['GET'])
def index_ex2():
    # 인덱스(이미지 뷰티)
    return render_template('index_ex2.html')

@app.route('/predict-img-cls', methods=['POST'])
def predict_cls():
    if request.method == 'POST':
        # 이미지 디코딩
        img = base64_to_pil(request.json)

        # 이미지 저장시 실행
        # img.save("./your save foloder/image.png")

        # 이미지 예측
        preds = img_cf.model_predict(img, img_cf.model)

        # 이미지 Max Probability
        pred_proba = "{:.3f}".format(np.amax(preds))

        # 이미지 분류 레이블
        pred_class = img_cf.decode_predictions(preds, top=1)

        # 분류 클래스 레이블 확인
        # print(pred_class)

        # 분류 레이블 출력
        result = str(pred_class[0][0][1])

        # 출력 보정
        result = result.replace('_', ' ').capitalize()

        # 이미지 리턴
        return jsonify(result=result, probability=pred_proba)

    return None

@app.route('/predict-img-beauty', methods=['POST'])
def predict_beauty():

    return make_response(jsonify(img_bt.model_predict()),200)

if __name__ == '__main__':
    # 일반 Flask Server Start
    # app.run(port=8080, threaded=False)

    # Gevent 서버 시작
    # 참고 : https://flask.palletsprojects.com/en/1.1.x/deploying/wsgi-standalone/#gevent
    http_server = WSGIServer(('0.0.0.0', 8080), app)
    http_server.serve_forever()
