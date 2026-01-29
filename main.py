from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import base64
from ultralytics import YOLO
import os

# =========================================================
# FASTAPI
# =========================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# SCHEMA
# =========================================================
class PredictRequest(BaseModel):
    image: str  # base64 limpio

# =========================================================
# PATHS
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "backend")

# =========================================================
# MODELOS
# =========================================================
modelrecorte = YOLO(os.path.join(MODELS_DIR, "recorte2.pt"))
modeldetOP   = YOLO(os.path.join(MODELS_DIR, "3clsOPfft.pt"))
modeldetOA   = YOLO(os.path.join(MODELS_DIR, "OAyoloR4cls5.pt"))

# =========================================================
# FUNCIONES
# =========================================================

def yolorecorte(model, img):
    results = model(img)
    coor = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            coor.append([x1, y1, x2, y2])
    return coor


def yolodetOPCrop(model, crop):
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    f = np.fft.fft2(crop)
    fshift = np.fft.fftshift(f)
    ms = 20 * np.log(np.abs(fshift) + 1)
    ms = ms.astype(np.uint8)

    results = model(ms)
    for r in results:
        cls = int(r.probs.top1)
        prob = float(r.probs.top1conf)

    return cls, prob


def yolodetOA(model, crop, certeza=0):
    results = model(crop)

    best_cls = None
    best_prob = 0
    best_box = None

    for result in results:
        for box in result.boxes:
            conf = box.conf[0].item()
            if conf > certeza and conf > best_prob:
                best_prob = conf
                best_cls = int(box.cls)
                best_box = tuple(map(int, box.xyxy[0]))

    if best_box is None:
        return None

    x1, y1, x2, y2 = best_box
    return best_cls, best_prob, x1, y1, x2, y2


def etiquetar(imagen, x1, y1, x2, y2, clOP, clOA=None, oa_box=None):
    etiquetas_op = ["OP: Normal", "OP: Osteopenia", "OP: Osteoporosis"]
    etiquetas_oa = ["OA: Normal", "OA: Dudoso", "OA: Leve", "OA: Moderado", "OA: Grave"]

    cv2.rectangle(imagen, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(imagen, etiquetas_op[clOP],
                (x1, y1 - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if clOA is not None and oa_box is not None:
        xa1, ya1, xa2, ya2 = oa_box
        p1 = (x1 + xa1, y1 + ya1)
        p2 = (x1 + xa2, y1 + ya2)

        cv2.rectangle(imagen, p1, p2, (0, 0, 255), 2)
        cv2.putText(imagen, etiquetas_oa[clOA],
                    (p1[0], p1[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return imagen


def unir_recortes(recortes):
    if len(recortes) == 0:
        return None

    max_width = max(r.shape[1] for r in recortes)
    resized = [
        cv2.resize(r, (max_width, int(r.shape[0] * max_width / r.shape[1])))
        for r in recortes
    ]
    return cv2.vconcat(resized)

# =========================================================
# ENDPOINT
# =========================================================
@app.post("/predict")
def predict(data: PredictRequest):
    try:
        img_bytes = base64.b64decode(data.image)
        np_img = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Imagen inválida")

        img_etiquetada = img.copy()
        recortes = yolorecorte(modelrecorte, img)

        if len(recortes) == 0:
            raise ValueError("No se detectaron rodillas")

        resultados = []
        crops_procesados = []

        for rec in recortes:
            x1, y1, x2, y2 = rec
            crop = img[y1:y2, x1:x2].copy()

            clOP, probOP = yolodetOPCrop(modeldetOP, crop)
            oa = yolodetOA(modeldetOA, crop)

            if oa:
                clOA, probOA, xa1, ya1, xa2, ya2 = oa
                oa_box = (xa1, ya1, xa2, ya2)
            else:
                clOA = probOA = oa_box = None

            img_etiquetada = etiquetar(
                img_etiquetada,
                x1, y1, x2, y2,
                clOP,
                clOA,
                oa_box
            )

            crops_procesados.append(crop)

            resultados.append({
                "clase_op": ["normal", "osteopenia", "osteoporosis"][clOP],
                "prob_op": probOP,
                "clase_oa": None if clOA is None else
                            ["normal", "oa-dudoso", "oa-leve", "oa-moderado", "oa-grave"][clOA],
                "prob_oa": probOA
            })

        img_procesada = unir_recortes(crops_procesados)

        _, buf_et = cv2.imencode(".jpg", img_etiquetada)
        _, buf_pr = cv2.imencode(".jpg", img_procesada)

        return {
            "resultado": resultados[0],  # frontend espera esto
            "imagenProcesada": "data:image/jpeg;base64," + base64.b64encode(buf_pr).decode(),
            "imagenEtiquetada": "data:image/jpeg;base64," + base64.b64encode(buf_et).decode(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
