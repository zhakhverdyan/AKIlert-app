import numpy as np
import pandas as pd
import json
import plotly
from flask import Flask, request, jsonify, render_template
#import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots


app = Flask(__name__)
#model = pickle.load(open('model.pkl', 'rb'))

#df_var = pd.read_csv('csv_files/var_subset.csv')
df_data = pd.read_csv('csv_files/for_app.csv')

"""
def get_prediction(num):
    df=pd.read_csv('csv_files/df_sample.csv')
    final_features = df.iloc[num-1].values.reshape(1,-1)
    prediction = np.squeeze(model.predict_proba(final_features))
    output = round(prediction[1], 3)
    adjusted_prob = output*0.5/0.03
    return round(adjusted_prob, 2)"""

def plot_figure(num, df_data):
    
    df = df_data[df_data['patient']==num]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    #fig = go.Figure()
    x=df['Lab measurement day'].values
    fig.add_trace(go.Scatter(
        x=df['Lab measurement day'].values,
        y=df['Normalized score'].values,
        name='Probability',
        mode='lines+markers'),
        secondary_y=False)


    fig.add_trace(
        go.Scatter(x=df['Lab measurement day'], y=df['creatinine'], name="Creatinine", mode='lines+markers'),
        secondary_y=True)

    fig.add_trace(
        go.Scatter(x=df['Lab measurement day'], y=[0.5]*len(df['Lab measurement day']), name="Threshold", fill="tozeroy", mode='lines',
        line_color='lightgrey'),
        secondary_y=False)
        

    fig.update_layout(
        title = 'Probability of acute kidney injury in the next 12-36h and creatinine levels',
        xaxis_title="Days in relation to ICU admission")

    fig.update_yaxes(title_text="Probability",range=[0,1], secondary_y=False)
    fig.update_yaxes(title_text="Creatinine (mg/dL)", secondary_y=True)
    fig.update_xaxes(range=[df['Lab measurement day'].min(), df['Lab measurement day'].max()])

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON


@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        print('posted')
        num=int(request.form.get("patient"))
        graphJSON=plot_figure(num, df_data)
        patient = request.form.get('patient', '')
        return render_template('index.html', patient = patient, graphJSON=graphJSON)
        #return render_template('index.html', patient = patient, prediction_text='Patient risk is {}'.format(get_prediction(num)), graphJSON=graphJSON)
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)