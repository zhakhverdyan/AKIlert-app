import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

def get_prediction(num):
    df=pd.read_csv('df_sample.csv')
    final_features = df.iloc[num-1].values.reshape(1,-1)
    prediction = np.squeeze(model.predict_proba(final_features))
    output = round(prediction[1], 3)
    return output


@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        print('posted')
        num=int(request.form.get("patient"))
        return render_template('index.html', prediction_text='Patient risk is {}'.format(get_prediction(num)))
    else:
        return render_template('index.html')

"""
def home():
    return render_template('index.html')



@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        num=int(request.form.get("patient"))
        return render_template('index.html', prediction_text='Patient risk is {}'.format(get_prediction(num)))
    else:
        return render_template('index.html')

@app.route('/results', methods=['GET', 'POST'])
def results():
    if request.method == "POST":

        data = request.get_json(force=True)
        prediction = model.predict([np.array(list(data.values()))])

        output = prediction[1]
        return jsonify(output)
        """

if __name__ == "__main__":
    app.run(debug=True)