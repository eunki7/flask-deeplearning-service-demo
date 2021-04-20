import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and models
from mlib import image_classfication as img_cf

# Any utilites
import numpy as np
from utils.util import base64_to_pil

# Declare a flask app
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index_ex1():
    # Main page
    return render_template('index_ex1.html')

@app.route('/beauty', methods=['GET'])
def index_ex2():
    # Main page
    return render_template('index_ex2.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Save the image to ./uploads
        # img.save("./uploads/image.png")

        # Make prediction
        preds = img_cf.model_predict(img, img_cf.model)

        # Process your result for human
        pred_proba = "{:.3f}".format(np.amax(preds))    # Max probability
        pred_class = img_cf.decode_predictions(preds, top=1)   # ImageNet Decode

        print(pred_class)

        result = str(pred_class[0][0][1])               # Convert to string
        result = result.replace('_', ' ').capitalize()

        # Serialize the result, you can add additional fields
        return jsonify(result=result, probability=pred_proba)

    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 8080), app)
    http_server.serve_forever()
