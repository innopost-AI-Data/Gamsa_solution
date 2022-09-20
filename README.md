# Gamsa_solution

## flask install
```
pip install flask
```

## server run
```
python server.py
```

### stt
inno_stt 경로안에 나스에서 다운받은 results 폴더 붙여넣기
```
\\172.30.1.24\데이터사업추진팀\01. 프로젝트\23. 2022년 AI바우처 지원사업\감사nlp\stt_model\results
```

### mrc
아래 경로의 **pytorch_model.bin** 파일을 inno_mrc\KorQuAD_Gamsa_model에 넣어야 함
```
Y:\01. 프로젝트\23. 2022년 AI바우처 지원사업\감사nlp\mrc_model\KorQuAD_Gamsa_models
```

### ner 모듈
|이름|버전|
|------|---|
|python|3.8.13|
|tensorflow|2.9.1|
|pytorch|1.11.0|
|cudatoolkit|11.3.1|
|cudnn|8.2.1|