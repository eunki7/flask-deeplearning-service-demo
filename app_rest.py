# Only flask
from flask import Flask, request, render_template, make_response, jsonify
from flask_restplus import Resource, Api, fields
from gevent.pywsgi import WSGIServer

# DL packages
from mlib import image_classfication as img_cf
from mlib import beauty_gan as img_bt
import numpy as np

# Utility
from utils.util import base64_to_pil, np_to_base64_bt

# Flask and Flask_restplus declare
app = Flask(__name__)

# Swagger gui setting
app.config.SWAGGER_UI_DOC_EXPANSION = 'list'

# Api initialize
api = Api(app, 
          # doc='/apidoc/',
          version='1.0', 
          title='Image DL API', 
          description='Images classfication and beauty GAN RestAPI'
    )

# Flask rest plus namespace define
predict_ns = api.namespace('predict', description='image prediction apis')

# Flask marshalling models
predict_ns_single_img = predict_ns.model(
    "Single input Image model",
    {
        "oriImage": fields.String(description="oriImage", required=True)
    },
)

predict_ns_two_img = predict_ns.model(
    "Two input Image model",
    {
        "oriImage": fields.String(description="oriImage", required=True),
        "mpImage": fields.String(description="mpImage", required=True)
    },
)

# Swagger Error Message
get_err_msg = { 
    200: 'Success',
    404: 'Page Not found',
}
post_err_msg = {
    200: 'Success',
    500: 'Wrong oriImg or mpImg parameters'
}

@api.route('/cls')
class ImageClsIndex(Resource):
    @api.doc(responses=get_err_msg)
    def get(self):
        headers = {'Content-Type': 'text/html'}
        return make_response(render_template('index.html'), 200, headers)
    
@api.route('/beauty')
class ImageBeautyIndex(Resource):
    @api.doc(responses=get_err_msg)
    def get(self):
        headers = {'Content-Type': 'text/html'}
        return make_response(render_template('index_beauty.html'), 200, headers)

@predict_ns.route('/img-cls', methods=['POST'])
class ImageClsPredict(Resource):
    @api.doc(responses=post_err_msg)
    @predict_ns.expect(predict_ns_single_img)
    def post(self):

        # Json request
        data = request.json

        # Image convert
        img = base64_to_pil(data.get('oriImage'))

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

@predict_ns.route('/img-beauty-single', methods=['POST'])
class ImageBeautyPredictSingle(Resource):
    @api.doc(responses=post_err_msg)
    @predict_ns.expect(predict_ns_single_img)
    def post(self):
        # Json request
        data = request.json
        
        # Image convert
        ori_img = base64_to_pil(data.get('oriImage'))
        
        # Test image save
        # ori_img.save("./image1.png")

        # Json response
        return jsonify(result=np_to_base64_bt(img_bt.predict_single_or_all(ori_img)))

@predict_ns.route('/img-beauty-all', methods=['POST'])
class ImageBeautyPredictAll(Resource):
    @api.doc(responses=post_err_msg)
    @predict_ns.expect(predict_ns_two_img)
    def post(self):
        # Json request
        data = request.json
        
        # Image convert
        ori_img, mp_img = base64_to_pil(data.get('oriImage')), base64_to_pil(data.get('mpImage'))
        
        # Test image save
        # ori_img.save("./image1.png")
        # mp_img.save("./image2.png")

        # Json response
        return jsonify(result=np_to_base64_bt(img_bt.predict_single_or_all(ori_img, mp_img)))

# RestApi errors handling
@predict_ns.errorhandler(Exception)
def predict_ns_handler(error):
    '''predict_ns error handler'''
    return {'message': 'Wrong oriImage or mpImage parameters.'}, getattr(error, 'code', 500)

if __name__ == '__main__':
    # Flask Server Start
    # app.run(port=8080, threaded=False)

    # Gevent Server Start
    # Refer : https://flask.palletsprojects.com/en/1.1.x/deploying/wsgi-standalone/#gevent
    http_server = WSGIServer(('0.0.0.0', 8080), app)
    http_server.serve_forever()
