from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import base64
from ultralytics import YOLO
import os

# ===============================
# CONFIG
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===============================
# 1) CARGA DE MODELOS
# ===============================
modelrecorte = YOLO(os.path.join(BASE_DIR, "backend", "recorte2.pt"))
modeldetOP   = YOLO(os.path.join(BASE_DIR, "backend", "3clsOPfft.pt"))
modeldetOA   = YOLO(os.path.join(BASE_DIR, "backend", "OAyoloR4cls5.pt"))

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
    image: str  # base64 limpio (sin data:image/...)

# ===============================
# FUNCIONES IA
# ===============================

def yolorecorte(model, img):
    results = model(img)
    coords = []

    for r in results:
        if r.boxes is None:
            continue
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            coords.append([x1, y1, x2, y2])

    return coords


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

    if r.boxes is None or len(r.boxes) == 0:
        return 0, 0.0, 0, 0, 0, 0

    b = r.boxes[0]
    x1, y1, x2, y2 = map(int, b.xyxy[0])
    return int(b.cls), float(b.conf[0]), x1, y1, x2, y2


def etiquetar(img, c, cl_op, prob_op, cl_oa, prob_oa, oa_box):
    x1, y1, x2, y2 = c

    # Caja ROI
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    etiquetas_op = ["Normal", "Osteopenia", "Osteoporosis"]
    cv2.putText(
        img,
        f"OP: {etiquetas_op[cl_op]} ({prob_op:.2f})",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # Caja OA (relativa al crop)
    xa1, ya1, xa2, ya2 = oa_box
    if xa2 > xa1 and ya2 > ya1:
        cv2.rectangle(
            img,
            (x1 + xa1, y1 + ya1),
            (x1 + xa2, y1 + ya2),
            (255, 0, 0),
            2
        )

        etiquetas_oa = ["Normal-dudoso", "OA-dudoso", "OA-leve", "OA-moderado", "OA-grave"]
        cv2.putText(
            img,
            f"OA: {etiquetas_oa[cl_oa]} ({prob_oa:.2f})",
            (x1 + xa1, y1 + ya1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

    return img


def procesar(img):
    coords = yolorecorte(modelrecorte, img)

    clase_op = "normal"
    clase_oa = "normal-dudoso"
    prob_op = prob_oa = 0.0

    img_out = img.copy()

    for c in coords:
        crop = img[c[1]:c[3], c[0]:c[2]]

        cl_op, prob_op = yolodetOPCrop(modeldetOP, crop)
        cl_oa, prob_oa, xa1, ya1, xa2, ya2 = yolodetOA(modeldetOA, crop)

        clase_op = ["normal", "osteopenia", "osteoporosis"][cl_op]
        clase_oa = ["normal-dudoso", "oa-dudoso", "oa-leve", "oa-moderado", "oa-grave"][cl_oa]

        img_out = etiquetar(
            img_out,
            c,
            cl_op,
            prob_op,
            cl_oa,
            prob_oa,
            (xa1, ya1, xa2, ya2)
        )

    return clase_op, prob_op, clase_oa, prob_oa, img_out


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
