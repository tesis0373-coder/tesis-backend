import os
import cv2
import base64
import numpy as np
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from functools import lru_cache

# ---------------------------
# App
# ---------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Rutas de modelos
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PATH_OP = os.path.join(BASE_DIR, "backend", "det 2cls R2 0.pt")
PATH_OA = os.path.join(BASE_DIR, "backend", "OAyoloIR4AH.pt")

# ---------------------------
# Cargar modelos una sola vez
# ---------------------------
@lru_cache(maxsize=1)
def load_models():
    print("🚀 Cargando modelos YOLO...")
    return YOLO(PATH_OP), YOLO(PATH_OA)

# ---------------------------
# Utilidades
# ---------------------------
def decode_base64_image(b64: str):
    if "," in b64:
        b64 = b64.split(",")[1]
    img_bytes = base64.b64decode(b64)
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Imagen corrupta")
    return img

def to_base64(img):
    _, buffer = cv2.imencode(".jpg", img)
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode()

# ---------------------------
# YOLO OP → recorte
# ---------------------------
def detect_op_and_crop(model, img, conf=0.0):
    results = model(img)
    best = None

    for r in results:
        for b in r.boxes:
            c = float(b.conf[0])
            if c >= conf:
                best = b

    if best is None:
        return None, None, None

    x1, y1, x2, y2 = map(int, best.xyxy[0])
    crop = img[y1:y2, x1:x2]
    cls = int(best.cls[0])
    prob = float(best.conf[0])

    return crop, cls, prob

# ---------------------------
# YOLO OA sobre recorte
# ---------------------------
def detect_oa(model, crop, conf=0.0):
    results = model(crop)
    best = None

    for r in results:
        for b in r.boxes:
            c = float(b.conf[0])
            if c >= conf:
                best = b

    if best is None:
        return None

    x1, y1, x2, y2 = map(int, best.xyxy[0])
    cls = int(best.cls[0])
    prob = float(best.conf[0])

    return cls, prob, x1, y1, x2, y2

# ---------------------------
# Etiquetado CORRECTO
# ---------------------------
def label_crop(crop, cls_op, cls_oa, box_oa):
    img = crop.copy()

    # OP siempre ocupa TODO el recorte
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (255, 0, 0), 2)
    text_op = "normal" if cls_op == 0 else "osteoporosis"
    cv2.putText(img, text_op, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # OA dentro del recorte
    if box_oa:
        cls, prob, x1, y1, x2, y2 = box_oa
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        if cls in [0, 1]:
            label = "normal-dudoso"
        elif cls in [2, 3]:
            label = "leve-moderado"
        else:
            label = "grave"

        cv2.putText(img, label, (x1 + 10, y1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return img

# ---------------------------
# API
# ---------------------------
@app.post("/predict")
async def predict(req: Request):
    data = await req.json()

    if "image" not in data:
        raise HTTPException(400, "No se recibió imagen")

    img = decode_base64_image(data["image"])

    model_op, model_oa = load_models()

    # 1️⃣ OP → recorte
    crop, cls_op, prob_op = detect_op_and_crop(model_op, img)
    if crop is None:
        raise HTTPException(400, "No se detectó rodilla")

    # 2️⃣ OA sobre recorte
    oa = detect_oa(model_oa, crop)

    # 3️⃣ Etiquetado
    crop_labeled = label_crop(crop, cls_op, oa[0] if oa else 0, oa)

    return {
        "resultado": {
            "clase_op": "normal" if cls_op == 0 else "osteoporosis",
            "prob_op": prob_op,
            "clase_oa":
                "normal-dudoso" if not oa or oa[0] in [0, 1]
                else "leve-moderado" if oa[0] in [2, 3]
                else "grave",
            "prob_oa": oa[1] if oa else 0.0
        },
        "imagenProcesada": to_base64(crop),
        "imagenEtiquetada": to_base64(crop_labeled)
    }

@app.get("/")
def health():
    return {"status": "ok"}
