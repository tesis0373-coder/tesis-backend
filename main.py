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
    coords = []
    for r in results:
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            coords.append([x1, y1, x2, y2])
    # ordenar izquierda → derecha
    return sorted(coords, key=lambda c: c[0])


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
        for b in r.boxes:
            conf = b.conf[0].item()
            if conf > certeza and conf > best_prob:
                best_prob = conf
                best = (
                    int(b.cls),
                    conf,
                    *map(int, b.xyxy[0])
                )

    return best


def etiquetar(img, op_cls, op_box, oa=None, idx=0):
    x1, y1, x2, y2 = op_box
    offset = idx * 30

    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    op_txt = ["normal", "osteopenia", "osteoporosis"][op_cls]
    cv2.putText(
        img,
        f"OP: {op_txt}",
        (x1, y1 - 10 - offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    if oa:
        cl, prob, xa1, ya1, xa2, ya2 = oa
        cv2.rectangle(
            img,
            (x1 + xa1, y1 + ya1),
            (x1 + xa2, y1 + ya2),
            (0, 0, 255),
            2
        )

        oa_txt = ["normal", "oa-dudoso", "oa-leve", "oa-moderado", "oa-grave"][cl]
        cv2.putText(
            img,
            f"OA: {oa_txt}",
            (x1 + xa1, y1 + ya1 - 10 - offset),
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
        img_bytes = base64.b64decode(data.image)
        np_img = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Imagen inválida")

        img_out = img.copy()
        coords = yolorecorte(modelrecorte, img)

        if not coords:
            raise ValueError("No se detectaron rodillas")

        resultados_op = []
        resultados_oa = []

        for idx, c in enumerate(coords):
            crop = img[c[1]:c[3], c[0]:c[2]].copy()

            clOP, probOP = yolodetOPCrop(modeldetOP, crop)
            oa = yolodetOA(modeldetOA, crop)

            img_out = etiquetar(img_out, clOP, c, oa, idx)

            resultados_op.append((clOP, probOP))
            if oa:
                resultados_oa.append((oa[0], oa[1]))

        # 🔥 RESUMEN GLOBAL (para el frontend)
        op_final = max(resultados_op, key=lambda x: x[0])
        if resultados_oa:
            oa_final = max(resultados_oa, key=lambda x: x[0])
        else:
            oa_final = (0, 0.0)

        _, buf = cv2.imencode(".jpg", img_out)

        return {
            "resultado": {
                "clase_op": ["normal", "osteopenia", "osteoporosis"][op_final[0]],
                "prob_op": op_final[1],
                "clase_oa": ["normal", "oa-dudoso", "oa-leve", "oa-moderado", "oa-grave"][oa_final[0]],
                "prob_oa": oa_final[1],
            },
            "imagenEtiquetada": "data:image/jpeg;base64," + base64.b64encode(buf).decode()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
