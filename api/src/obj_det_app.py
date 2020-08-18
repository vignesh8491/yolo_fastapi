import sys
import time
import uuid
sys.path.append("./src")

from fastapi import FastAPI, File
import numpy as np
from starlette.requests import Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

import config as cfg
from engine import Engine, RemoteEngine
from tool import resp_util

if cfg.APP_MODE == 0:
    engine = Engine()
else:
    engine = RemoteEngine()


app = FastAPI()
app.mount("/out", StaticFiles(directory="out"), name="out")

@app.get("/home")
def home():

    resp_content = resp_util.get_home_response()
    return HTMLResponse(content=resp_content)


@app.post("/predict")
async def predict(request: Request,
            file: bytes = File(...)):
    

    if request.method == "POST":

        session_id = uuid.uuid1()

        t1 = time.time()
        out_path = engine.predict(file, session_id)
        
        # get HTML response content after prediction
        resp_content = resp_util.get_prediction_response(out_path)
        t2=time.time()

        print("----------")
        print('total latency:')
        print(str((t2-t1)*1000)+' ms')
        print("----------")

    return HTMLResponse(resp_content)
