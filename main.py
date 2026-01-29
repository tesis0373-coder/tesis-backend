from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import base64
from ultralytics import YOLO
import os

# ======================================================
# FASTAPI
# ======================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================
# SCHEMA
# ======================================================
class PredictRequest(BaseModel):
    image: str  # base64 limpio (SIN data:image/...)

# ======================================================
# PATHS
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "backend")

# ======================================================
# MODELOS
# ======================================================
modelrecorte = YOLO(os.path.join(MODELS_DIR, "recorte2.pt"))
modeldetOP   = YOLO(os.path.join(MODELS_DIR, "3clsOPfft.pt"))
modeldetOA   = YOLO(os.path.join(MODELS_DIR, "OAyoloR4cls5.pt"))

# ======================================================
# FUNCIONES
# ======================================================

# -------- Recorte de rodilla --------
def yolorecorte(model, img):
    results = model(img)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            return x1, y1, x2, y2
    return None


# -------- Osteoporosis (FFT + clasificador) --------
def yolodetOPCrop(model, crop):
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    f = np.fft.fft2(crop)
    fshift = np.fft.fftshift(f)
    mag = 20 * np.log(np.abs(fshift) + 1)
    mag = mag.astype(np.uint8)

    results = model(mag)
    for r in results:
        cls = int(r.probs.top1)
        prob = float(r.probs.top1conf)
        return cls, prob

    return 0, 0.0


# -------- Osteoartritis (YOLO detección) --------
def yolodetOA(model, crop, certeza=0.0):
    results = model(crop)

    best_cls = None
    best_prob = 0
    best_box = None

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf > certeza and conf > best_prob:
                best_prob = conf
                best_cls = int(box.cls)
                best_box = tuple(map(int, box.xyxy[0]))

    if best_box is None:
        return None

    x1, y1, x2, y2 = best_box
    return best_cls, best_prob, x1, y1, x2, y2


# -------- Etiquetar imagen --------
def etiquetar(imagen, x1, y1, x2, y2,
              clOP,
              clOA=None, xa1=None, ya1=None, xa2=None, ya2=None):

    img = imagen.copy()

    # ---- OP ----
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    op_label = ["Normal", "Osteopenia", "Osteoporosis"][clOP]
    cv2.putText(
        img,
        f"OP: {op_label}",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # ---- OA ----
    if clOA is not None:
        p1 = (x1 + xa1, y1 + ya1)
        p2 = (x1 + xa2, y1 + ya2)

        cv2.rectangle(img, p1, p2, (0, 0, 255), 2)

        oa_label = ["Normal", "OA dudoso", "OA leve", "OA moderado", "OA grave"][clOA]
        cv2.putText(
            img,
            f"OA: {oa_label}",
            (p1[0], p1[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

    return img


# ======================================================
# ENDPOINT
# ======================================================
@app.post("/predict")
def predict(data: PredictRequest):
    try:
        # ---------- Decode base64 ----------
        img_bytes = base64.b64decode(data.image)
        np_img = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Imagen inválida")

        # ---------- Recorte ----------
        rec = yolorecorte(modelrecorte, img)
        if rec is None:
            raise ValueError("No se detectó rodilla")

        x1, y1, x2, y2 = rec
        crop = img[y1:y2, x1:x2].copy()

        # ---------- Inferencias ----------
        clOP, probOP = yolodetOPCrop(modeldetOP, crop)

        oa = yolodetOA(modeldetOA, crop)
        if oa:
            clOA, probOA, xa1, ya1, xa2, ya2 = oa
        else:
            clOA = probOA = xa1 = ya1 = xa2 = ya2 = None

        # ---------- Imágenes ----------
        imgProcesada = crop.copy()
        imgEtiquetada = etiquetar(
            img, x1, y1, x2, y2,
            clOP,
            clOA, xa1, ya1, xa2, ya2
        )

        # ---------- Encode ----------
        _, buf_proc = cv2.imencode(".jpg", imgProcesada)
        _, buf_etq  = cv2.imencode(".jpg", imgEtiquetada)

        return {
            "resultado": {
                "clase_op": ["normal", "osteopenia", "osteoporosis"][clOP],
                "prob_op": probOP,
                "clase_oa": None if clOA is None else
                            ["normal", "oa-dudoso", "oa-leve", "oa-moderado", "oa-grave"][clOA],
                "prob_oa": probOA,
            },
            "imagenProcesada": "data:image/jpeg;base64," + base64.b64encode(buf_proc).decode(),
            "imagenEtiquetada": "data:image/jpeg;base64," + base64.b64encode(buf_etq).decode(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
