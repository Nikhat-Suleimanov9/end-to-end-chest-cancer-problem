from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from cnnChestCancer.utils.common import decodeImage
from cnnChestCancer.pipeline.predict import PredictionPipeline
import json

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classification = PredictionPipeline(self.filename)


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')



@app.route("/train", methods=['GET','POST'])
@cross_origin()
def trainRoute():
    os.system("python main.py")
    #os.system("dvc repro")
    return "Training done successfully!"




@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    image = request.json['image']
    decodeImage(image,cApp.filename)
    result = cApp.classification.predict()
    return jsonify(result)


@app.route("/scores", methods=["GET"])
def get_scores():
    scores_path = os.path.join("scores.json")

    if not os.path.exists(scores_path):
        return jsonify({"error": "scores.json not found"}), 404

    with open(scores_path, "r") as f:
        scores = json.load(f)

    return jsonify(scores)

if __name__ == "__main__":
    cApp = ClientApp()
    app.run(host='0.0.0.0', port=8080) #for AWS