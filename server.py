from flask import Flask, render_template, request, jsonify
from inno_ocr.main import run
import os
from werkzeug.utils import secure_filename
import numpy as np
import librosa
import soundfile as sf
import ffmpeg
import base64
import uuid
import json
from inno_stt.recognizer import Recognizer

stt_recognizer = Recognizer(output_dir='./inno_stt/logs',
                        model_cfg='../tasks/SpeechRecognition/kconfspeech/configs/jasper10x5dr_sp_offline_specaugment.yaml',
                        ckpt='../tasks/SpeechRecognition/kconfspeech/results/Jasper_epoch95_checkpoint.pt',
                        task_path="../tasks.SpeechRecognition.kconfspeech.local.manifest",
                        vocab="../tasks/SpeechRecognition/kconfspeech/data/KconfSpeech/vocab",
                        decoding_mode='ctc_decoder',
#                        decoding_mode='',
                        lm_path="../tasks/SpeechRecognition/kconfspeech/data/KconfSpeech/5gram_korean.binary"
                        )
stt_recognizer.load_model()


app = Flask(__name__)

@app.route('/')
def homepage():
    return render_template("index.html")

@app.route('/stt')
def stt():
    return render_template("stt.html")
@app.route('/inference_stt', methods=['POST'])
def inference_stt():
    data = request.get_json()
    dec_data = base64.b64decode(data["data"])
    return ""
@app.route('/ner')
def ner():
    return render_template("index.html")

@app.route('/mrc')
def mrc():
    return render_template("index.html")

@app.route('/ocr')
def ocr():
    return render_template("ocr.html")

@app.route('/inference_ocr', methods=['POST'])
def inference_ocr():
    if 'file' not in request.files:
        print('No file part')
        return jsonify(result=render_template("result_file.html", filename = "", result=""))
    else:
        file = request.files.get('file')
        if not os.path.exists('./inno_ocr/test/'):
            os.makedirs('./inno_ocr/test/')
        file.save('./inno_ocr/test/' + file.filename)
        target = []
        target_dir = './inno_ocr/test/'
        target.append(target_dir + file.filename)
        result = run(target)
    return jsonify(result=render_template("result_file.html", filename = file.filename, result=result))

@app.route('/buttons')
def buttons():
    return render_template("buttons.html")
@app.route('/cards')
def cards():
    return render_template("cards.html")

@app.route('/utilities-color')
def color():
    return render_template("utilities-color.html")
@app.route('/utilities-border')
def border():
    return render_template("utilities-border.html")
@app.route('/utilities-animation')
def animation():
    return render_template("utilities-animation.html")
@app.route('/utilities-other')
def other():
    return render_template("utilities-other.html")

@app.route('/login')
def login():
    return render_template("login.html")
@app.route('/register')
def register():
    return render_template("register.html")
@app.route('/forgot_password')
def forgot_password():
    return render_template("forgot-password.html")
@app.route('/not_found')
def not_found():
    return render_template("404.html")
@app.route('/blank')
def blank():
    return render_template("blank.html")

@app.route('/charts')
def charts():
    return render_template("charts.html")
@app.route('/tables')
def tables():
    return render_template("tables.html")


if __name__ == "__main__":
    app.run(debug=True)