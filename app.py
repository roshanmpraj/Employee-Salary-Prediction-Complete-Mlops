

from fastapi import FastAPI

import pandas as pd

import joblib

app = FastAPI()

model = joblib.load("model.pkl")

encoder = joblib.load("encoder.pkl")

@app.get("/")

def home():

    return {

        "message":"Salary Prediction API Running"

    }

@app.get("/predict")

def predict(

    age:int,

    experience:int,

    education:str,

    skillscore:int

):

    education = encoder.transform(

        [education]

    )[0]

    data = pd.DataFrame(

        {

            "Age":[age],

            "Experience":[experience],

            "Education":[education],

            "SkillScore":[skillscore]

        }

    )

    prediction = model.predict(

        data

    )

    return {

        "Predicted Salary":int(prediction[0])

    }

