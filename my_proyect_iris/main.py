from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn
import os

# Cargar modelo
modelo = joblib.load("model/modelo.pkl")
clases = ["Setosa", "Versicolor", "Virginica"]

# Inicializar FastAPI
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Ruta principal con formulario
@app.get("/", response_class=HTMLResponse)
async def form_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Procesar predicción desde formulario HTML
@app.post("/predict-html", response_class=HTMLResponse)
async def predict_html(
    request: Request,
    sepal_length: float = Form(...),
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...)
):
    datos = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    pred = modelo.predict(datos)[0]
    clase = clases[pred]
    return templates.TemplateResponse("resultado.html", {"request": request, "clase": clase})

# (Opcional) Predicción con JSON
class IrisEntrada(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict-json")
def predict_api(entrada: IrisEntrada):
    datos = np.array([[entrada.sepal_length, entrada.sepal_width,
                       entrada.petal_length, entrada.petal_width]])
    pred = modelo.predict(datos)[0]
    return {"clase_predicha": clases[pred]}

