from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import base64
from ultralytics import YOLO
import os

# ===============================
# PATHS
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "backend")

# ===============================
# MODELOS
# ===============================
modelrecorte = YOLO(os.path.join(MODELS_DIR, "recorte2.pt"))
modeldetOP   = YOLO(os.path.join(MODELS_DIR, "3clsOPfft.pt"))
modeldetOA   = YOLO(os.path.join(MODELS_DIR, "OAyoloR4cls5.pt"))

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
# FUNCIONES
# ===============================
def yolorecorte(model, img):
    r = model(img)[0]
    b = r.boxes[0]
    x1, y1, x2, y2 = map(int, b.xyxy[0])
    return x1, y1, x2, y2


def yolodetOPCrop(model, crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    ms = 20 * np.log(np.abs(fshift) + 1)
    ms = ms.astype(np.uint8)

    r = model(ms)[0]
    return int(r.probs.top1), float(r.probs.top1conf)


def yolodetOA(model, crop):
    r = model(crop)[0]
    b = r.boxes[0]
    return int(b.cls), float(b.conf), *map(int, b.xyxy[0])


def etiquetar(img, clOP, clOA, oa_box):
    etiquetas_op = ["Normal", "Osteopenia", "Osteoporosis"]
    etiquetas_oa = ["Normal", "OA dudoso", "OA leve", "OA moderado", "OA grave"]

    cv2.putText(img, etiquetas_op[clOP], (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    x1, y1, x2, y2 = oa_box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.putText(img, etiquetas_oa[clOA], (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return img


# ===============================
# ENDPOINT
# ===============================
@app.post("/predict")
def predict(data: PredictRequest):
    try:
        # -------- decodificar imagen --------
        img_bytes = base64.b64decode(data.image)
        np_img = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Imagen inválida")

        # -------- RECORTE --------
        x1, y1, x2, y2 = yolorecorte(modelrecorte, img)
        crop = img[y1:y2, x1:x2].copy()

        # -------- OP / OA --------
        clOP, probOP = yolodetOPCrop(modeldetOP, crop)
        clOA, probOA, xa1, ya1, xa2, ya2 = yolodetOA(modeldetOA, crop)

        # -------- imagenes --------
        img_procesada = crop.copy()
        img_etiquetada = etiquetar(crop.copy(), clOP, clOA, (xa1, ya1, xa2, ya2))

        # -------- encode --------
        _, buf_proc = cv2.imencode(".jpg", img_procesada)
        _, buf_etq  = cv2.imencode(".jpg", img_etiquetada)

        return {
            "resultado": {
                "clase_op": ["normal", "osteopenia", "osteoporosis"][clOP],
                "prob_op": probOP,
                "clase_oa": ["normal", "oa-dudoso", "oa-leve", "oa-moderado", "oa-grave"][clOA],
                "prob_oa": probOA,
            },
            "imagenProcesada": "data:image/jpeg;base64," + base64.b64encode(buf_proc).decode(),
            "imagenEtiquetada": "data:image/jpeg;base64," + base64.b64encode(buf_etq).decode(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
