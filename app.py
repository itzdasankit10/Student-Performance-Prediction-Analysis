from fastapi import FastAPI, Form
import pandas as pp
import joblib as jb
from fastapi.middleware.cors import CORSMiddleware

##CREATING FASTAPI SERVER
app = FastAPI()
model = jb.load("model.pkl")
model_cols = jb.load("model_cols.pkl")


app.add_middleware(
    CORSMiddleware,
    allow_origins = ["http://127.0.0.1:3000"],
    allow_credentials = True,
    allow_methods=["*"],
    allow_headers=["*"],
)

##TAKING THR VALUES ENTERED IN THE FORM 
@app.post("/predict")
def predict(
    Gender: str = Form(...),
    EthnicGroup: str = Form(...),
    ParentEdc: str = Form(...),
    LunchType: str = Form(...),
    TestPrep: str = Form(...),
    ParentMaritalStatus: str= Form(...),
    PracticeSprt: str = Form(...),
    IsFirstChild: str = Form(...),
    TransportMeans: str = Form(...),
    WklyStudyHours: str = Form(...),
    NrSibling: int = Form(...)
):
    data = {
        "Gender":Gender,
        "EthnicGroup":EthnicGroup,
        "ParentEduc":ParentEdc,
        "LunchType":LunchType,
        "TestPrep":TestPrep,
        "ParentMaritalStatus":ParentMaritalStatus,
        "PracticeSport":PracticeSprt,
        "IsFirstChild":IsFirstChild,
        "TransportMeans":TransportMeans,
        "WklyStudyHours":WklyStudyHours,
        "NrSiblings":NrSibling
    }
    datas = pp.DataFrame([data])
    datas = pp.get_dummies(datas)
    datas = datas.reindex(columns=model_cols, fill_value=0)

    prediction = model.predict(datas)[0]
    return {"prediction":prediction}