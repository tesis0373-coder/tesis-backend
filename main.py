from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import base64
from ultralytics import YOLO
import os
# ===============================
# FASTAPI
# ===============================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# SCHEMA
# ===============================
class PredictRequest(BaseModel):
    image: str  # base64 limpio

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
# FUNCIONES
# ===============================

# -------- Detectar rodillas --------
def yolorecorte(modelrecorte, img):
    results = modelrecorte(img)
    coor = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            coor.append([x1, y1, x2, y2])
    return coor


# -------- Detectar osteoporosis (FFT) --------
def yolodetOPCrop(modeldetOPfft, crop):

    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    f = np.fft.fft2(crop)
    fshift = np.fft.fftshift(f)
    ms = 20 * np.log(np.abs(fshift) + 1)
    ms = ms.astype(np.uint8)

    results = modeldetOPfft(ms)

    for result in results:
        cls = int(result.probs.top1)
        prob = float(result.probs.top1conf)

    return cls, prob


# -------- Detectar osteoartritis (YOLO) --------
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


# -------- Etiquetar imagen --------
def etiquetar2(imagen,
               clOP, xOP1, yOP1, xOP2, yOP2,
               clOA=None, xOA1=None, yOA1=None, xOA2=None, yOA2=None):

    color = (255, 0, 0)
    grosor = 2

    # ---- OP ----
    cv2.rectangle(imagen, (xOP1, yOP1), (xOP2, yOP2), color, grosor)

    if clOP == 0:
        etiqueta = 'Sin osteoporosis'
    elif clOP == 1:
        etiqueta = 'Osteopenia'
    else:
        etiqueta = 'Osteoporosis'

    cv2.putText(imagen, etiqueta,
                (xOP1, yOP1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 255, 0), 2, cv2.LINE_AA)

    # ---- OA ----
    if clOA is not None:
        p1 = (xOP1 + xOA1, yOP1 + yOA1)
        p2 = (xOP1 + xOA2, yOP1 + yOA2)

        cv2.rectangle(imagen, p1, p2, color, grosor)

        if clOA == 0:
            etiqueta = 'Sin Osteoartrosis'
        elif clOA == 1:
            etiqueta = 'OA dudoso'
        elif clOA == 2:
            etiqueta = 'OA leve'
        elif clOA == 3:
            etiqueta = 'OA moderado'
        else:
            etiqueta = 'OA grave'

        cv2.putText(imagen, etiqueta,
                    (p1[0], p1[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0), 2, cv2.LINE_AA)

    return imagen


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

        img_etiquetada = img.copy()

        # -------- DETECCIÓN DE RODILLAS --------
        recortes = yolorecorte(modelrecorte, img)

        if len(recortes) == 0:
            raise ValueError("No se detectaron rodillas")

        resultados = []

        for rec in recortes:
            x1, y1, x2, y2 = rec
            crop = img[y1:y2, x1:x2].copy()

            # OP
            clOP, probOP = yolodetOPCrop(modeldetOP, crop)

            # OA
            oa = yolodetOA(modeldetOA, crop, certeza=0)

            if oa:
                clOA, probOA, xa1, ya1, xa2, ya2 = oa
            else:
                clOA = probOA = xa1 = ya1 = xa2 = ya2 = None

            # Etiquetar (ESTA ES LA CLAVE)
            img_etiquetada = etiquetar2(
                img_etiquetada,
                clOP, x1, y1, x2, y2,
                clOA, xa1, ya1, xa2, ya2
            )

            resultados.append({
                "clase_op": ["normal", "osteopenia", "osteoporosis"][clOP],
                "prob_op": probOP,
                "clase_oa": None if clOA is None else
                            ["normal", "oa-dudoso", "oa-leve", "oa-moderado", "oa-grave"][clOA],
                "prob_oa": probOA
            })

        # -------- encode --------
        _, buf = cv2.imencode(".jpg", img_etiquetada)

        return {
            "resultados": resultados,
            "imagenEtiquetada": "data:image/jpeg;base64," + base64.b64encode(buf).decode()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
