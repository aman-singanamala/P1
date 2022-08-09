from fastapi import FastAPI
import uvicorn
import pickle
from infos import AAAA


app= FastAPI()

pickle_in= open("model.pkl","rb")
classifier = pickle.load(pickle_in)

@app.get('/')
def home():
    return {'Hello User': '!'}
@app.post('predict')
def predict_diabetes(data: AAAA):
    data= data.dict()
    print(data)
    Pregnancies= data['Pregnancies']
    Glucose: data['Glucose']
    BloodPressure: data['BloodPressure']
    SkinThickness: data['SkinThickness']
    Insulin : data['Insulin']
    BMI : data['BMI']
    DiabetesPedigreeFunction: data['DiabetesPedigreeFunction']
    Age: data['Age']
    prediction= classifier.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,
    BMI,DiabetesPedigreeFunction,Age]])

    if(prediction[0]==1):
        prediction="Have Diabetes"
    else:
        prediction="Do not have diabetes"
    return{
        'prediction':prediction
    }
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
