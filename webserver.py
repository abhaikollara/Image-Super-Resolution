import main
import json
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)


@app.route('/sr', methods=["POST"])
def sr():
    image = json.loads(request.data)['image']
    sr_image = main.web_request(image)
    data = {'image': sr_image}
    return jsonify(data)
