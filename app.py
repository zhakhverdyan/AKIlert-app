import numpy as np
import pandas as pd
import csv
from collections import OrderedDict
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    d = OrderedDict()
    with open('column_names.csv') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            for name in row:
                d[name] = 0

    # numerical features, update to actual values
    num_feat = ["WBC x 1000", "bicarbonate", "chloride", "creatinine", "platelets x 1000", "potassium",\
         "sodium", "age", "unitvisitnumber"]
    for feat in num_feat:
        value = request.form[feat]
        d[feat] = value

    # categorical features, update selected values to 1
    cat_feat = ["sex", "ethnicity", "unittype", "unitadmitsource"]
    for feat in cat_feat:
        value = request.form[feat]
        key_value = feat+'_'+value
        if key_value in d.keys():
            d[key_value] = 1

    record = pd.DataFrame(dict(d), columns=list(d.keys()), index=[0])
    final_features = record.values
    prediction = np.squeeze(model.predict_proba(final_features.reshape(1, -1)))

    output = round(prediction[1], 3)

    return render_template('index.html', prediction_text='Patient risk is {}'.format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[1]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)