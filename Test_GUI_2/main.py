from flask import Flask, render_template, request
import bag_of_words

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/handle_data', methods=['GET', 'POST'])
def handle_data():
    url = request.form['inputURL']
    question = request.form['inputQuestion']
    answer = bag_of_words.get_answer(url, question)

    return render_template('handle_data.html', result_url=url, result_question=question,
                           result_answer=answer)

if __name__ == '__main__':
    app.run(debug=True)
