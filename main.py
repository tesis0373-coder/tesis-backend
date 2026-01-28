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
    """Detecta TODAS las rodillas"""
    results = model(img)
    coords = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            coords.append((x1, y1, x2, y2))
    return coords


def yolodetOPCrop(model, crop):
    """Clasificación OP con FFT"""
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    f = np.fft.fft2(crop)
    fshift = np.fft.fftshift(f)
    ms = 20 * np.log(np.abs(fshift) + 1)
    ms = ms.astype(np.uint8)

    r = model(ms)[0]
    return int(r.probs.top1), float(r.probs.top1conf)


def yolodetOA(model, crop, certeza=0.0):
    """Detección OA (mejor bounding box)"""
    results = model(crop)
    best = None

    for r in results:
        for box in r.boxes:
            conf = box.conf[0].item()
            if conf > certeza:
                if best is None or conf > best["conf"]:
                    best = {
                        "cls": int(box.cls),
                        "conf": conf,
                        "xyxy": tuple(map(int, box.xyxy[0]))
                    }

    if best is None:
        return None

    return best["cls"], best["conf"], *best["xyxy"]


def etiquetar(img, recorte, clOP, clOA, oa_box):
    """Dibuja cajas y etiquetas correctamente"""
    x1, y1, x2, y2 = recorte

    etiquetas_op = ["Normal", "Osteopenia", "Osteoporosis"]
    etiquetas_oa = ["Normal", "OA dudoso", "OA leve", "OA moderado", "OA grave"]

    # --- OP (usa caja del recorte)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(
        img,
        etiquetas_op[clOP],
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    # --- OA (remapeo local → global)
    if clOA is not None and oa_box is not None:
        xa1, ya1, xa2, ya2 = oa_box
        p1 = (x1 + հոդվածxa1, y1 + ya1)
        p2 = (x1 + xa2, y1 + ya2)

        cv2.rectangle(img, p1, p2, (0, 0, 255), 2)
        cv2.putText(
            img,
            etiquetas_oa[clOA],
            (p1[0], p1[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

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
            oa = yolodetOA(modeldetOA, crop)
            if oa:
                clOA, probOA, xa1, ya1, xa2, ya2 = oa
                oa_box = (xa1, ya1, xa2, ya2)
            else:
                clOA, probOA, oa_box = None, None, None

            # Etiquetar
            img_etiquetada = etiquetar(
                img_etiquetada,
                rec,
                clOP,
                clOA,
                oa_box
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
