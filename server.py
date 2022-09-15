## ocr
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
from inno_mrc.model import main

# stt_recognizer = Recognizer(output_dir='./inno_stt/logs',
#                         model_cfg='../tasks/SpeechRecognition/kconfspeech/configs/jasper10x5dr_sp_offline_specaugment.yaml',
#                         ckpt='../tasks/SpeechRecognition/kconfspeech/results/Jasper_epoch95_checkpoint.pt',
#                         task_path="../tasks.SpeechRecognition.kconfspeech.local.manifest",
#                         vocab="../tasks/SpeechRecognition/kconfspeech/data/KconfSpeech/vocab",
#                         decoding_mode='ctc_decoder',
# #                        decoding_mode='',
#                         lm_path="../tasks/SpeechRecognition/kconfspeech/data/KconfSpeech/5gram_korean.binary"
#                         )
# stt_recognizer.load_model()

## ner
# 변수 + 토크나이저 + 모델 불러오기 + gpu 할당 설정
import tensorflow as tf
import numpy as np

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

from inno_ner import tokenization_kobert
from transformers import TFBertModel

tokenizer = tokenization_kobert.KoBertTokenizer.from_pretrained('monologg/kobert')
# new_model = tf.keras.models.load_model("inno_ner/kobert_tf2crf_all_es10",custom_objects={"TFBertModel":TFBertModel.from_pretrained("monologg/kobert", from_pt=True)})

# 라벨 사전
index_to_ner = {0: '-', 1: 'AC_B', 2: 'AC_I', 3: 'CT_B', 4: 'CT_I', 5: 'DR_B', 6: 'DR_I', 7: 'DT_B', 8: 'DT_I', 9: 'EV_B', 10: 'EV_I', 11: 'LC_B', 12: 'LC_I', 13: 'MY_B', 14: 'MY_I', 15: 'NOG_B', 16: 'NOG_I', 17: 'OG_B', 18: 'OG_I', 19: 'QT_B', 20: 'QT_I', 21: 'TI_B', 22: 'TI_I', 23: 'TX_B', 24: 'TX_I', 25: '[PAD]'}

# 문장길이
max_len = 178

## mrc

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

@app.route('/ner',methods=["GET","POST"])
def ner():
    if request.method == "POST":
        sentences = request.form.get("sentences")

        def ner_inference(test_sentence): # 문장 집어넣기
            result = []
            # 테스트 문장, 마스크 설정
            global tokenized_sentence, tokenized_mask
            
            tokenized_sentence = np.array([tokenizer.encode(test_sentence, max_length=max_len, truncation=True, padding='max_length')])
            tokenized_mask = np.array([[int(x!=1) for x in tokenized_sentence[0].tolist()]])
            # 모델에 투입
            # ans = new_model.predict([tokenized_sentence, tokenized_mask])
            tokens = tokenizer.convert_ids_to_tokens(tokenized_sentence[0])
            
            new_tokens, new_labels = [], []
            for token, label_idx in zip(tokens, ans[0]):
                if (token.startswith("▁")): # _으로 시작하면
                    if token == "▁":
                        pass
                    else:
                        new_labels.append(index_to_ner[label_idx])
                        new_tokens.append(token[1:]) # ##뒤에 단어만 추가되게 설정
                # 문장시작,끝,패딩 은 넘어가기
                elif (token=='[CLS]'): 
                    pass
                elif (token=='[SEP]'):
                    pass
                elif (token=='[PAD]'):
                    pass
                # 앞, 뒤 토큰이 아니면 추가
                elif (token != '[CLS]' or token != '[SEP]'):
                    new_tokens.append(token)
                    new_labels.append(index_to_ner[label_idx])
            # 토큰, 라벨을 동시에 리스트에 저장
            for token, label in zip(new_tokens, new_labels):
                result.append([token, label])
            return result

        # 토큰,라벨 동시에 저장
        aa = ner_inference(sentences)
        
        # 개체명에 해당하는 단어들을 하나로 묶어 저장
        ans = []
        ans_text = ""
        for i,x in enumerate(aa):
            if x[1] != "-": # 개체명이 있다면
                if aa[i-1][1][:-2] == aa[i][1][:-2]: # 이전 개체명이랑 현재 개체명이랑 같다면
                    ans_text += x[0]
                elif aa[i-1][1][:-2] != aa[i][1][:-2]: #이전 개체명이랑 현재 개체명이 다르다면
                    ans.append(str(aa[i-1][1][:-2]+" : "+ ans_text))
                    ans_text = ""
                    ans_text += x[0]
            elif x[1] == "-":
                ans.append(str(aa[i-1][1][:-2]+" : "+ ans_text))
                ans_text = ""
        # 마지막 토큰까지 개체명인 경우 elif에 해당하지 않아서 ans에 추가되지 않으므로 개별로 설정
        if ans_text != "":
            ans.append(str(aa[i-1][1][:-2]+" : "+ ans_text))
            ans_text = ""

        # " : " 요소를 제거하자
        ans2 = []
        for x in ans:
            if x != " : ":
                ans2.append(x)

        # 영어 개체명에 해당하는 한국어 개체명으로 바꿔서 보이기

        bb = sentences

        print("입력하신 문장 :",bb)
        print("태깅 데이터셋 :",aa)

        return render_template("ner.html",
            files = aa,
            sen = bb,
            ans = ans2,
        )
    return render_template("ner.html")

@app.route('/mrc')
def mrc():
    return render_template("mrc.html")

@app.route("/mrc_inference", methods=['GET', 'POST'])
def mrc_inference():
    if request.method == 'POST':
        query = request.form["query"]
        result = main(query)
    return render_template("mrc_result.html", query=query, result=result)

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