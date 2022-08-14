from crypt import methods
from inspect import classify_class_attrs
from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
classifier= pickle.load(open('model.pkl','rb'))

app= Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])

def predict():
    if request.method=='POST':
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trtbps = int(request.form['trtbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalachh = int(request.form['thalachh'])
        exng = int(request.form['exng'])
        oldpeak = int(request.form['oldpeak'])
        slp = int(request.form['slp'])
        caa = int(request.form['caa'])
        thall = int(request.form['thall'])

        data= np.array([[age,sex,cp,trtbps,chol,fbs,restecg,thalachh,exng,oldpeak,slp,caa,thall]])
        
        my_prediction= classifier.predict(data)


        return render_template('result.html',prediction=my_prediction)


if __name__=='__main__': # THIS IS FOR LOADING 
    app.run(debug=True)