import os
import sys

# Only flask
from flask import Flask, request, render_template, Response, jsonify
from gevent.pywsgi import WSGIServer

# DL packages
from mlib import image_classfication as img_cf
from mlib import beauty_gan as img_bt

import numpy as np

# Utility
from utils.util import base64_to_pil, np_to_base64_bt

# Flask declare
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index_ex1():
    # Main test page : Image Classfication
    return render_template('index_ex1.html')

@app.route('/beauty', methods=['GET'])
def index_ex2():
    # Main test page : BeautyGan
    return render_template('index_ex2.html')

@app.route('/predict-img-cls', methods=['POST'])
def predict_cls():
    
    # Image convert
    img = base64_to_pil(request.json)

    # Test image save
    # img.save("./your save foloder/image.png")

    # Image predict
    preds = img_cf.predict(img, img_cf.model)

    # Value : Image max probability
    pred_proba = "{:.3f}".format(np.amax(preds))

    # Label : Image classfication
    pred_class = img_cf.decode_predictions(preds, top=1)

    # Check image label
    # print(pred_class)

    # Result : image label
    result = str(pred_class[0][0][1])

    # label capitalize
    result = result.replace('_', ' ').capitalize()

    # Json response
    return jsonify(result=result, probability=pred_proba)


@app.route('/predict-img-beauty-single', methods=['POST'])
def predict_beauty_single():

    # Json request
    data = request.json
    
    # Image convert
    ori_img = base64_to_pil(data.get('oriImage'))
    
    # Test image save
    # ori_img.save("./image1.png")

    # Json response
    return jsonify(result=np_to_base64_bt(img_bt.predict_single_or_all(ori_img)))

@app.route('/predict-img-beauty-all', methods=['POST'])
def predict_beauty_all():

    # Json request
    data = request.json
    
    # Image convert
    ori_img, mp_img = base64_to_pil(data.get('oriImage')), base64_to_pil(data.get('mpImage'))
    
    # Test image save
    # ori_img.save("./image1.png")
    # mp_img.save("./image2.png")

    # Json response
    return jsonify(result=np_to_base64_bt(img_bt.predict_single_or_all(ori_img, mp_img)))


if __name__ == '__main__':
    # Flask Server Start
    # app.run(port=8080, threaded=False)

    # Gevent Server Start
    # Refer : https://flask.palletsprojects.com/en/1.1.x/deploying/wsgi-standalone/#gevent
    http_server = WSGIServer(('0.0.0.0', 8080), app)
    http_server.serve_forever()
