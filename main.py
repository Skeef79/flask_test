from flask import Flask,redirect,url_for,render_template,request
import os

import cv2
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import efficientnet.tfkeras
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import numpy as np

img_width = 192
img_height = 128
pathologies = ['bacterial_spot',
               'black_rot',
               'cedar_apple_rust',
               'cercospora_leaf_spot',
               'common_rust',
               'early_blight',
               'esca',
               'haunglongbing',
               'healthy',
               'late_blight',
               'leaf_blight',
               'leaf_mold',
               'leaf_scorch',
               'northern_leaf_blight',
               'powdery_mildew',
               'scab',
               'septoria_leaf_spot',
               'spider_mites',
               'target_spot',
               'tomato_mosaic_virus',
               'tomato_yellow_leaf_curl_virus'
               ]

def load_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if (img.shape[0] > img.shape[1]):
        img = np.rot90(img)

    img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_AREA)

    img = img.astype('float32')
    img -= img.min()
    img /= img.max()

    return img


def load_model(model_path, weights_path):
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weights_path)

    print("model loaded successfully")
    return loaded_model

model = load_model('model/model.json','model/model_weights.h5')

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/",methods=["POST","GET"])
def home():
    return render_template("index.html")

@app.route("/upload",methods=["POST","GET"])
def upload():
    if request.method=="POST":
        target = os.path.join(APP_ROOT,'images')
        print(target)

        destination = None

        for file in request.files.getlist("file"):
            print(file)
            filename = file.filename
            if filename =="":
                return redirect(url_for("home"))

            destination = "/".join([target,filename])
            print(destination)
            file.save(destination)

        img = load_img(destination).reshape(-1, img_height, img_width, 3)
        y_pred = model.predict(img)
        res = pathologies[y_pred.argmax()]
        print(y_pred)
        return redirect(url_for("complete",result=res))
    else:
        return render_template("index.html")

@app.route("/<result>")
def complete(result):
    return render_template('complete.html',content=result)

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    application.run(debug=False, port=port, host='0.0.0.0', threaded=True)
    