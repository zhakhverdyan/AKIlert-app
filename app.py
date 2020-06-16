import numpy as np
import pandas as pd
import json
import plotly
from flask import Flask, request, jsonify, render_template
import pickle
import plotly.graph_objects as go

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

df_var = pd.read_csv('csv_files/var_subset.csv')

def get_prediction(num):
    df=pd.read_csv('csv_files/df_sample.csv')
    final_features = df.iloc[num-1].values.reshape(1,-1)
    prediction = np.squeeze(model.predict_proba(final_features))
    output = round(prediction[1], 3)
    adjusted_prob = output*0.5/0.03
    return round(adjusted_prob, 2)

def plot_figure(num):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_var[df_var['aki_label']==0]['min_result_calcium'],\
                            y=df_var[df_var['aki_label']==0]['max_result_creatinine'], \
                        mode='markers',
                        name='negative class'))
    fig.add_trace(go.Scatter(x=df_var[df_var['aki_label']==1]['min_result_calcium'],\
                            y=df_var[df_var['aki_label']==1]['max_result_creatinine'], \
                        mode='markers',
                        name='positive class'))
    fig.add_trace(go.Scatter(x=[5], y=[30],
                        mode='markers',
                        name='hypothetical patient'))
    fig.update_layout(title='Creatinine and calcium levels by patient class',
                    xaxis_title="Minimum Calcium level (mg/dL)",
                    yaxis_title="Maximum Creatinine level (mg/dL)",)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON


@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        print('posted')
        num=int(request.form.get("patient"))
        graphJSON=plot_figure(num)
        patient = request.form.get('patient', '')
        print(patient)
        return render_template('index.html', patient = patient, prediction_text='Patient risk is {}'.format(get_prediction(num)), graphJSON=graphJSON)
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)