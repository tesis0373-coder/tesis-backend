from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import base64
import os
from ultralytics import YOLO

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

def yolorecorte(model, img):
    results = model(img)
    cajas = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cajas.append([x1, y1, x2, y2])
    return cajas


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

    return 0, 0.0


def yolodetOA(model, crop, certeza=0):
    results = model(crop)

    best = None
    best_prob = 0

    for r in results:
        for box in r.boxes:
            conf = box.conf[0].item()
            if conf > certeza and conf > best_prob:
                best_prob = conf
                best = (
                    int(box.cls),
                    conf,
                    *map(int, box.xyxy[0])
                )

    return best


def etiquetar2(img, x1, y1, x2, y2, clOP, clOA=None, boxOA=None):
    # Caja general
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    texto_op = ["Sin osteoporosis", "Osteopenia", "Osteoporosis"][clOP]
    cv2.putText(
        img,
        f"OP: {texto_op}",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    if clOA is not None and boxOA is not None:
        xa1, ya1, xa2, ya2 = boxOA
        cv2.rectangle(
            img,
            (x1 + xa1, y1 + ya1),
            (x1 + xa2, y1 + ya2),
            (0, 0, 255),
            2
        )

        texto_oa = [
            "Normal",
            "OA dudoso",
            "OA leve",
            "OA moderado",
            "OA grave"
        ][clOA]

        cv2.putText(
            img,
            f"OA: {texto_oa}",
            (x1 + xa1, y1 + ya1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

    return img

# ===============================
# ENDPOINT
# ===============================
@app.post("/predict")
def predict(data: PredictRequest):
    try:
        # ---- Decodificar imagen ----
        img_bytes = base64.b64decode(data.image)
        np_img = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Imagen inválida")

        img_etiquetada = img.copy()

        # ---- Detectar rodillas ----
        rodillas = yolorecorte(modelrecorte, img)

        if len(rodillas) == 0:
            raise ValueError("No se detectaron rodillas")

        # ---- IMAGEN PROCESADA (CLAVE) ----
        if len(rodillas) == 1:
            x1, y1, x2, y2 = rodillas[0]
        else:
            x1 = min(r[0] for r in rodillas)
            y1 = min(r[1] for r in rodillas)
            x2 = max(r[2] for r in rodillas)
            y2 = max(r[3] for r in rodillas)

        imagen_procesada = img[y1:y2, x1:x2].copy()

        resultados = []

        # ---- Analizar cada rodilla ----
        for r in rodillas:
            rx1, ry1, rx2, ry2 = r
            crop = img[ry1:ry2, rx1:rx2].copy()

            clOP, probOP = yolodetOPCrop(modeldetOP, crop)
            oa = yolodetOA(modeldetOA, crop)

            if oa:
                clOA, probOA, xa1, ya1, xa2, ya2 = oa
                boxOA = (xa1, ya1, xa2, ya2)
            else:
                clOA = probOA = boxOA = None

            img_etiquetada = etiquetar2(
                img_etiquetada,
                rx1, ry1, rx2, ry2,
                clOP,
                clOA,
                boxOA
            )

            resultados.append({
                "clase_op": ["normal", "osteopenia", "osteoporosis"][clOP],
                "prob_op": probOP,
                "clase_oa": None if clOA is None else
                            ["normal", "oa-dudoso", "oa-leve", "oa-moderado", "oa-grave"][clOA],
                "prob_oa": probOA
            })

        # ---- Encode imágenes ----
        _, buf_proc = cv2.imencode(".jpg", imagen_procesada)
        _, buf_et = cv2.imencode(".jpg", img_etiquetada)

        return {
            "resultado": resultados[0],  # frontend espera uno
            "imagenProcesada": "data:image/jpeg;base64," + base64.b64encode(buf_proc).decode(),
            "imagenEtiquetada": "data:image/jpeg;base64," + base64.b64encode(buf_et).decode()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
