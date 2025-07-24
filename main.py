import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import io
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
import json


model = keras.models.load_model('model-trFalse-0.8850.h5',
                                custom_objects={'KerasLayer': hub.KerasLayer})

label = ["caries", "discoloration"]

app = Flask(__name__)

def predict_label(img):
    i = np.asarray(img) / 255.0
    i = i.reshape(1, 224, 224, 3)
    pred = model.predict(i)
    result = pred[0].tolist()
    # output = {str(i):j for i,j in zip(label,result)}
    output = {str(i):f"{j*100:.2f}%" for i, j in zip(label, result)}

    return output

@app.route("/predict", methods=["GET","POST"])
def index():
    file = request.files.get('file')
    if file is None or file.filename == "":
        return jsonify({"error" : "no file"})

    image_bytes = file.read()
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((224,224), Image.NEAREST)
    pred_img = predict_label(img)
    return pred_img
    

@app.route("/", methods=["GET"])
def home():
    return "Hello World! This belongs to the route path of Multilabel Classification API"

if __name__ == "__main__":
    # app.run(debug=True)
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))