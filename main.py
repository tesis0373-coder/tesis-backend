from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import cv2
import numpy as np
from ultralytics import YOLO
import os

# --------------------------------------------------
# 🚀 APP
# --------------------------------------------------
app = FastAPI()

# --------------------------------------------------
# 🌐 CORS (OBLIGATORIO)
# --------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5000",
        "http://127.0.0.1:5000",
        "https://tesis-app.web.app",
        "https://tesis-app.firebaseapp.com",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# 🧠 CARGA DE MODELOS YOLO
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")

try:
    model_recorte = YOLO(os.path.join(BACKEND_DIR, "corte0.pt"))
    model_op = YOLO(os.path.join(BACKEND_DIR, "det 2cls R2 0.pt"))
    model_oa = YOLO(os.path.join(BACKEND_DIR, "OAyoloIR4AH.pt"))
except Exception as e:
    print("❌ ERROR cargando modelos:", e)
    raise e

# --------------------------------------------------
# 📦 SCHEMA
# --------------------------------------------------
class PredictRequest(BaseModel):
    image: str  # Base64 completo: data:image/png;base64,...

# --------------------------------------------------
# 🧪 UTILIDADES
# --------------------------------------------------
def decode_base64_image(data: str) -> np.ndarray:
    try:
        if "," in data:
            data = data.split(",")[1]

        img_bytes = base64.b64decode(data)
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Imagen inválida")

        return img
    except Exception:
        raise HTTPException(status_code=400, detail="Imagen corrupta o inválida")

def encode_image(img: np.ndarray) -> str:
    _, buffer = cv2.imencode(".jpg", img)
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode()

# --------------------------------------------------
# 🔬 ENDPOINT PRINCIPAL
# --------------------------------------------------
@app.post("/predict")
def predict(req: PredictRequest):

    # 1️⃣ Decodificar imagen
    img = decode_base64_image(req.image)

    # 2️⃣ RECORTE (si aplica)
    recorte_result = model_recorte(img)[0]
    if len(recorte_result.boxes) > 0:
        box = recorte_result.boxes[0].xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box
        img_crop = img[y1:y2, x1:x2]
    else:
        img_crop = img.copy()

    # 3️⃣ DETECCIÓN OSTEOPOROSIS
    res_op = model_op(img_crop)[0]
    clase_op = "normal"
    prob_op = 0.0

    if len(res_op.boxes) > 0:
        box = res_op.boxes[0]
        prob_op = float(box.conf[0])
        clase_op = "osteoporosis"

    # 4️⃣ DETECCIÓN OSTEOARTRITIS
    res_oa = model_oa(img_crop)[0]
    clase_oa = "normal"
    prob_oa = 0.0

    if len(res_oa.boxes) > 0:
        box = res_oa.boxes[0]
        prob_oa = float(box.conf[0])
        clase_oa = "osteoartritis"

    # 5️⃣ Imágenes de salida
    img_procesada = encode_image(img_crop)
    img_etiquetada = encode_image(img_crop)  # aquí puedes dibujar cajas si quieres

    # 6️⃣ RESPUESTA
    return {
        "resultado": {
            "clase_op": clase_op,
            "prob_op": round(prob_op, 3),
            "clase_oa": clase_oa,
            "prob_oa": round(prob_oa, 3),
        },
        "imagenProcesada": img_procesada,
        "imagenEtiquetada": img_etiquetada,
    }

# --------------------------------------------------
# 🟢 HEALTH CHECK
# --------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok"}
