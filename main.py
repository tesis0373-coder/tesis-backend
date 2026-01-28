from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import base64
from ultralytics import YOLO
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===============================
# MODELOS
# ===============================
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
# FUNCIONES IA
# ===============================
def yolorecorte(model, img):
    results = model(img)
    coords = []
    for r in results:
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            coords.append([x1, y1, x2, y2])
    return coords


def procesar_fft(crop):
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    f = np.fft.fft2(crop)
    fshift = np.fft.fftshift(f)
    ms = 20 * np.log(np.abs(fshift) + 1)
    ms = ms.astype(np.uint8)

    return ms


def yolodetOP(model, img_fft):
    r = model(img_fft)[0]
    return int(r.probs.top1), float(r.probs.top1conf)


def yolodetOA(model, crop):
    r = model(crop)[0]
    box = r.boxes[0]
    return int(box.cls), float(box.conf), box.xyxy[0]


def etiquetar(img, op_cls, oa_cls, x1, y1, x2, y2, oa_box):
    etiquetas_op = ["Normal", "Osteopenia", "Osteoporosis"]
    etiquetas_oa = ["Normal-dudoso", "OA dudoso", "OA leve", "OA moderado", "OA grave"]

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(img, etiquetas_op[op_cls], (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    xa1, ya1, xa2, ya2 = map(int, oa_box)
    cv2.rectangle(img, (xa1, ya1), (xa2, ya2), (255, 0, 0), 2)
    cv2.putText(img, etiquetas_oa[oa_cls], (xa1, ya1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return img


# ===============================
# PIPELINE
# ===============================
def pipeline(img):
    coords = yolorecorte(modelrecorte, img)

    if not coords:
        raise ValueError("No se detectó ROI")

    # Tomamos el primer recorte (o el último, como prefieras)
    x1, y1, x2, y2 = coords[0]
    crop = img[y1:y2, x1:x2]

    # Imagen procesada (FFT)
    img_procesada = procesar_fft(crop)

    # Predicciones
    op_cls, op_prob = yolodetOP(modeldetOP, img_procesada)
    oa_cls, oa_prob, oa_box = yolodetOA(modeldetOA, crop)

    # Imagen etiquetada (COPIA de la procesada)
    img_etiquetada = cv2.cvtColor(img_procesada, cv2.COLOR_GRAY2BGR)
    img_etiquetada = etiquetar(
        img_etiquetada,
        op_cls,
        oa_cls,
        0, 0, img_etiquetada.shape[1], img_etiquetada.shape[0],
        oa_box
    )

    return {
        "op": op_cls,
        "op_prob": op_prob,
        "oa": oa_cls,
        "oa_prob": oa_prob,
        "procesada": img_procesada,
        "etiquetada": img_etiquetada
    }


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

        res = pipeline(img)

        _, buf_p = cv2.imencode(".jpg", res["procesada"])
        _, buf_e = cv2.imencode(".jpg", res["etiquetada"])

        return {
            "resultado": {
                "clase_op": ["normal", "osteopenia", "osteoporosis"][res["op"]],
                "prob_op": res["op_prob"],
                "clase_oa": ["normal-dudoso", "oa-dudoso", "oa-leve", "oa-moderado", "oa-grave"][res["oa"]],
                "prob_oa": res["oa_prob"],
            },
            "imagenProcesada": f"data:image/jpeg;base64,{base64.b64encode(buf_p).decode()}",
            "imagenEtiquetada": f"data:image/jpeg;base64,{base64.b64encode(buf_e).decode()}",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
