# from django.shortcuts import render
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
import pickle


app=Flask(__name__)

regmodel=pickle.load(open('model.pkl','rb'))
scalar=pickle.load(open('scaler.pkl','rb'))


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])

def predict_api():
    data=request.json['data']
    print(data)
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])


if __name__=="__main__":
    app.run(debug=True)