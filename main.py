import os
import cv2
import numpy as np
import base64
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

base_path = os.path.dirname(__file__)

path_detOP = os.path.join(base_path, "backend", "det 2cls R2 0.pt")
path_detOA = os.path.join(base_path, "backend", "OAyoloIR4AH.pt")

@lru_cache(maxsize=1)
def load_models():
    print(">>> Cargando modelos")
    return YOLO(path_detOP), YOLO(path_detOA)

class ImageData(BaseModel):
    image: str

@app.post("/predict")
async def predict(payload: ImageData):
    print(">>> predict llamado")

    modeldetOP, modeldetOA = load_models()

    image_b64 = payload.image.split(",")[-1]

    try:
        image_bytes = base64.b64decode(image_b64)
    except Exception:
        raise HTTPException(400, "Base64 inválido")

    nparr = np.frombuffer(image_bytes, np.uint8)
    img_original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_original is None:
        raise HTTPException(400, "Imagen ilegible")

    certeza = 0

    clOP, prOP, crop, x1, y1, x2, y2 = yolodetOPCrop(modeldetOP, img_original, certeza)

    if crop is None:
        raise HTTPException(400, "No se detectó OP")

    clOA, prOA, xa1, ya1, xa2, ya2 = yolodetOA(modeldetOA, crop, certeza)

    return {"ok": True}

@app.get("/")
def health():
    return {"status": "ok"}
