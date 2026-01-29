from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import base64
from ultralytics import YOLO
import os

# ==========================================================
# FASTAPI
# ==========================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================
# SCHEMA
# ==========================================================
class PredictRequest(BaseModel):
    image: str  # base64 limpio (sin data:image/jpeg;base64,)

# ==========================================================
# PATHS
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "backend")

# ==========================================================
# MODELOS
# ==========================================================
modelrecorte = YOLO(os.path.join(MODELS_DIR, "recorte2.pt"))
modeldetOP   = YOLO(os.path.join(MODELS_DIR, "3clsOPfft.pt"))
modeldetOA   = YOLO(os.path.join(MODELS_DIR, "OAyoloR4cls5.pt"))

# ==========================================================
# FUNCIONES
# ==========================================================

# ---------------------------
# Detectar rodillas
# ---------------------------
def yolorecorte(model, img):
    results = model(img)
    coor = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            coor.append([x1, y1, x2, y2])

    return coor


# ---------------------------
# Detectar osteoporosis (FFT + YOLO)
# ---------------------------
def yolodetOPCrop(modeldetOPfft, crop):

    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # FFT
    f = np.fft.fft2(crop)
    fshift = np.fft.fftshift(f)
    ms = 20 * np.log(np.abs(fshift) + 1)
    ms = ms.astype(np.uint8)

    results = modeldetOPfft(ms)

    for result in results:
        cls = int(result.probs.top1)
        prob = float(result.probs.top1conf)

    return cls, prob


# ---------------------------
# Detectar osteoartritis (YOLO)
# ---------------------------
def yolodetOA(modeldet, crop, certeza=0):

    results = modeldet(crop)

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


# ---------------------------
# Etiquetar imagen (OP + OA)
# ---------------------------
def etiquetar2(
    imagen,
    clOP, xOP1, yOP1, xOP2, yOP2,
    clOA=None, xOA1=None, yOA1=None, xOA2=None, yOA2=None
):

    color_op = (255, 0, 0)
    color_oa = (0, 0, 255)
    grosor = 2

    # ---- OP ----
    cv2.rectangle(imagen, (xOP1, yOP1), (xOP2, yOP2), color_op, grosor)

    if clOP == 0:
        etiqueta = "Sin osteoporosis"
    elif clOP == 1:
        etiqueta = "Osteopenia"
    else:
        etiqueta = "Osteoporosis"

    cv2.putText(
        imagen,
        etiqueta,
        (xOP1, yOP1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # ---- OA ----
    if clOA is not None:
        p1 = (xOP1 + xOA1, yOP1 + yOA1)
        p2 = (xOP1 + xOA2, yOP1 + yOA2)

        cv2.rectangle(imagen, p1, p2, color_oa, grosor)

        if clOA == 0:
            etiqueta = "Sin OA"
        elif clOA == 1:
            etiqueta = "OA dudoso"
        elif clOA == 2:
            etiqueta = "OA leve"
        elif clOA == 3:
            etiqueta = "OA moderado"
        else:
            etiqueta = "OA grave"

        cv2.putText(
            imagen,
            etiqueta,
            (p1[0], p1[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return imagen


# ==========================================================
# ENDPOINT
# ==========================================================
@app.post("/predict")
def predict(data: PredictRequest):

    # RESPUESTA SEGURA POR DEFECTO (para el frontend)
    respuesta_segura = {
        "clase_op": None,
        "prob_op": None,
        "clase_oa": None,
        "prob_oa": None,
        "resultados": [],
        "imagenEtiquetada": None
    }

    try:
        # -------- decodificar imagen --------
        img_bytes = base64.b64decode(data.image)
        np_img = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None:
            return respuesta_segura   # 👈 NO explota el frontend

        img_etiquetada = img.copy()
        recortes = yolorecorte(modelrecorte, img)

        if not recortes:
            return respuesta_segura   # 👈 frontend seguro

        resultados = []

        for rec in recortes:
            x1, y1, x2, y2 = rec
            crop = img[y1:y2, x1:x2].copy()

            clOP, probOP = yolodetOPCrop(modeldetOP, crop)
            oa = yolodetOA(modeldetOA, crop)

            img_etiquetada = etiquetar2(
                img_etiquetada,
                clOP, x1, y1, x2, y2,
                None if oa is None else oa[0],
                None if oa is None else oa[2],
                None if oa is None else oa[3],
                None if oa is None else oa[4],
                None if oa is None else oa[5],
            )

            resultados.append({
                "clase_op": ["normal", "osteopenia", "osteoporosis"][clOP],
                "prob_op": probOP,
                "clase_oa": None if oa is None else
                    ["normal", "oa-dudoso", "oa-leve", "oa-moderado", "oa-grave"][oa[0]],
                "prob_oa": None if oa is None else oa[1]
            })

        if not resultados:
            return respuesta_segura

        resumen = resultados[0]
        _, buf = cv2.imencode(".jpg", img_etiquetada)

        return {
            **resumen,
            "resultados": resultados,
            "imagenEtiquetada": "data:image/jpeg;base64," + base64.b64encode(buf).decode()
        }

    except Exception as e:
        # 👇 JAMÁS rompas el frontend
        return respuesta_segura
