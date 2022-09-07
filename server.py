from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def homepage():
    return render_template("index.html")

@app.route('/stt')
def stt():
    return render_template("index.html")

@app.route('/ner')
def ner():
    return render_template("index.html")

@app.route('/mrc')
def mrc():
    return render_template("index.html")

@app.route('/ocr')
def ocr():
    return render_template("index.html")

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