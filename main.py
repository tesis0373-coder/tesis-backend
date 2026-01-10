from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import numpy as np
import cv2
from ultralytics import YOLO
import io
import os

# -------------------------
# App
# -------------------------
app = FastAPI(title="Backend Radiografías - YOLO")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # luego puedes restringir
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Cargar modelos (UNA VEZ)
# -------------------------
try:
    model_recorte = YOLO("backend/corte0.pt")
    model_op = YOLO("backend/det 2cls R2 0.pt")
    model_oa = YOLO("backend/OAyoloIR4AH.pt")
except Exception as e:
    print("❌ Error cargando modelos:", e)
    raise e

# -------------------------
# Utils
# -------------------------
def decode_base64_image(b64_string: str) -> np.ndarray:
    try:
        if "," in b64_string:
            b64_string = b64_string.split(",")[1]

        img_bytes = base64.b64decode(b64_string)
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Imagen inválida")

        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error decodificando imagen: {e}")

# -------------------------
# Schemas
# -------------------------
class Base64Request(BaseModel):
    image_base64: str

# -------------------------
# Endpoints
# -------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "Backend activo"}

@app.post("/predict")
async def predict(
    file: UploadFile = File(None),
    data: Base64Request | None = None
):
    # 1️⃣ Obtener imagen
    if file:
        contents = await file.read()
        img_array = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Archivo de imagen inválido")

    elif data:
        img = decode_base64_image(data.image_base64)

    else:
        raise HTTPException(status_code=400, detail="No se recibió imagen")

    # 2️⃣ Inferencia YOLO
    try:
        r_crop = model_recorte(img, verbose=False)
        r_op = model_op(img, verbose=False)
        r_oa = model_oa(img, verbose=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en inferencia: {e}")

    # 3️⃣ Respuesta simple (puedes extender)
    return {
        "osteoporosis": len(r_op[0].boxes) > 0,
        "osteoartritis": len(r_oa[0].boxes) > 0,
        "detecciones_op": len(r_op[0].boxes),
        "detecciones_oa": len(r_oa[0].boxes),
    }
