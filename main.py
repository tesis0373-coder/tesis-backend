import os
import cv2
import base64
import numpy as np
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

# ---------------------------
# APP
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
# MODELOS
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")

modelrecorte = YOLO(os.path.join(BACKEND_DIR, "recorte2.pt"))
modeldetOP = YOLO(os.path.join(BACKEND_DIR, "3clsOPfft.pt"))
modeldetOA = YOLO(os.path.join(BACKEND_DIR, "OAyoloR4cls5.pt"))

# ---------------------------
# UTILIDADES
# ---------------------------
def decode_base64_image(b64):
    if "," in b64:
        b64 = b64.split(",")[1]
    arr = np.frombuffer(base64.b64decode(b64), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Imagen inválida")
    return img

def to_base64(img):
    _, buffer = cv2.imencode(".jpg", img)
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode()

# ---------------------------
# RECORTE (IGUAL A TU MAIN)
# ---------------------------
def yolorecorte(model, img):
    results = model(img)
    coor = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            coor.append([x1, y1, x2, y2])
    return coor

# ---------------------------
# PIPELINE BASE
# ---------------------------
def CorrerModelo(img):
    coor = yolorecorte(modelrecorte, img)

    if not coor:
        raise HTTPException(status_code=400, detail="No se detectó rodilla")

    # 🔹 Si hay 1 → ese crop
    # 🔹 Si hay 2 → crop que cubra ambas
    x1 = min(c[0] for c in coor)
    y1 = min(c[1] for c in coor)
    x2 = max(c[2] for c in coor)
    y2 = max(c[3] for c in coor)

    crop = img[y1:y2, x1:x2].copy()

    # Normalización ligera (solo visual)
    procesada = cv2.normalize(crop, None, 0, 255, cv2.NORM_MINMAX)

    return procesada

# ---------------------------
# API
# ---------------------------
@app.post("/predict")
async def predict(req: Request):
    data = await req.json()

    if "image" not in data:
        raise HTTPException(status_code=400, detail="No se recibió imagen")

    img = decode_base64_image(data["image"])
    procesada = CorrerModelo(img)

    return {
        "imagenProcesada": to_base64(procesada),
        "imagenEtiquetada": None,
        "resultado": None
    }

@app.get("/")
def health():
    return {"status": "ok"}
