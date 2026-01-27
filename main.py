from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import base64
from ultralytics import YOLO
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ---------------------------
# 1) CARGA DE MODELOS
# ---------------------------
modelrecorte = YOLO(os.path.join(BASE_DIR, "backend", "recorte2.pt"))
modeldetOP = YOLO(os.path.join(BASE_DIR, "backend", "3clsOPfft.pt"))
modeldetOA = YOLO(os.path.join(BASE_DIR, "backend", "OAyoloR4cls5.pt"))

# ===============================
# FASTAPI
# ===============================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# SCHEMA
# ===============================
class PredictRequest(BaseModel):
    image: str  # base64 limpio

# ===============================
# FUNCIONES IA (LAS TUYAS)
# ===============================
def yolorecorte(model, img):
    results = model(img)
    coor = []
    for r in results:
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            coor.append([x1, y1, x2, y2])
    return coor


def yolodetOPCrop(model, crop):
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    f = np.fft.fft2(crop)
    fshift = np.fft.fftshift(f)
    ms = 20 * np.log(np.abs(fshift) + 1)
    ms = ms.astype(np.uint8)

    r = model(ms)[0]
    return int(r.probs.top1), float(r.probs.top1conf)


def yolodetOA(model, crop):
    r = model(crop)[0]
    box = r.boxes[0]
    return int(box.cls), float(box.conf)


def procesar(img):
    coords = yolorecorte(modelrecorte, img)

    clase_op = "normal"
    clase_oa = "normal-dudoso"
    prob_op = prob_oa = 0.0

    for c in coords:
        crop = img[c[1]:c[3], c[0]:c[2]]

        op, prob_op = yolodetOPCrop(modeldetOP, crop)
        oa, prob_oa = yolodetOA(modeldetOA, crop)

        clase_op = ["normal", "osteopenia", "osteoporosis"][op]
        clase_oa = ["normal-dudoso", "oa-dudoso", "oa-leve", "oa-moderado", "oa-grave"][oa]

    return clase_op, prob_op, clase_oa, prob_oa, img


# ===============================
# ENDPOINT
# ===============================
@app.post("/predict")
def predict(data: PredictRequest):
    try:
        img_bytes = base64.b64decode(data.image)
        np_img = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Imagen inválida")

        clase_op, prob_op, clase_oa, prob_oa, img_out = procesar(img)

        _, buffer = cv2.imencode(".jpg", img_out)
        img_b64 = base64.b64encode(buffer).decode("utf-8")

        return {
            "resultado": {
                "clase_op": clase_op,
                "prob_op": prob_op,
                "clase_oa": clase_oa,
                "prob_oa": prob_oa,
            },
            "imagenProcesada": f"data:image/jpeg;base64,{img_b64}",
            "imagenEtiquetada": f"data:image/jpeg;base64,{img_b64}",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
