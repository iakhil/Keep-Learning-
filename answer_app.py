import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
ans = pickle.load(open('answer_model.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/quest')
def show_ques():
    
    return render_template('index.html')

@app.route('/answer')
def render_ans():
    return render_template('answer.html')
    



@app.route('/answer',methods=['POST'])
def answer():
    '''
    For rendering results on HTML GUI
    '''
    print("Running answer_app.py")
    formid = request.args.get('formid', 1, type=int)
    input_text = [x for x in request.form.values()]
    print(input_text)
    # final_features = [np.array(int_features)]
    # prediction = model.predict(final_features)

    # output = round(prediction[0], 2)

    output = ans({
        'question': input_text[1], 
        'context': input_text[0]
    })

    passage_text = input_text[0]

    return render_template('answer.html', passage_text=' {}'.format(passage_text), prediction_text='Answer: {}'.format(output))

# @app.route('/predict_api',methods=['POST'])
# # def predict_api():
# #     '''
# #     For direct API calls trought request
# #     '''
# #     data = request.get_json(force=True)
# #     prediction = model.predict([np.array(list(data.values()))])

# #     output = prediction[0]
# #     return jsonify(output)

@app.route('/quest',methods=['POST'])
def quest():
    '''
    For rendering results on HTML GUI
    '''
    input_text = [x for x in request.form.values()]
    input_text = input_text[0]
    print(input_text) 
    # final_features = [np.array(int_features)]
    preds = model(input_text)
    print(preds)
    questions = []
    answers = []
    for i in range(len(preds)):
        questions.append(str(i + 1) + '. ' + preds[i]['question'])
        answers.append(str(i + 1) + '. ' + preds[i]['answer'])
    sep = ", "
    return render_template('index.html', prediction_text='{}'.format("\n".join(questions)), answer_text='{}'.format("\n".join(answers)))

if __name__ == "__main__":
    app.run(debug=False)