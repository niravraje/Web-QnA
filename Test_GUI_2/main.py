from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def home():
    render_template("home.html")

@app.route("/page2")
def page2():
    return "Hi, this is the 2nd page."

if __name__ == "__main__":
    app.run(debug=True)

