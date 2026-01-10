import os
import cv2
import numpy as np
import base64
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from functools import lru_cache

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
BASE_DIR = os.path.dirname(__file__)
PATH_CORTE = os.path.join(BASE_DIR, "backend", "corte0.pt")
PATH_OP = os.path.join(BASE_DIR, "backend", "det 2cls R2 0.pt")
PATH_OA = os.path.join(BASE_DIR, "backend", "OAyoloIR4AH.pt")

@lru_cache(maxsize=1)
def load_models():
    print(">>> Cargando modelos YOLO...")
    return (
        YOLO(PATH_CORTE),
        YOLO(PATH_OP),
        YOLO(PATH_OA),
    )

# ---------------------------
# Utils
# ---------------------------
def decode_base64_image(img_str: str):
    if "," in img_str:
        img_str = img_str.split(",")[-1]

    img_bytes = base64.b64decode(img_str)
    img_np = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Imagen inválida")

    return img

def to_base64(img):
    _, buf = cv2.imencode(".jpg", img)
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()

# ---------------------------
# Endpoint
# ---------------------------
@app.post("/predict")
async def predict(request: Request):
    corte_model, op_model, oa_model = load_models()

    data = await request.json()
    if "image" not in data:
        return {"error": "No se recibió la imagen"}

    img = decode_base64_image(data["image"])
    img_draw = img.copy()

    resultados = []

    # 1️⃣ Detectar rodillas
    cortes = corte_model(img)[0]

    for box in cortes.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img[y1:y2, x1:x2]

        # OP
        op_res = op_model(crop)[0]
        op_cls, op_prob = None, 0.0
        if len(op_res.boxes) > 0:
            b = max(op_res.boxes, key=lambda b: b.conf[0])
            op_cls = int(b.cls)
            op_prob = float(b.conf)

        # OA
        oa_res = oa_model(crop)[0]
        oa_cls, oa_prob = None, 0.0
        if len(oa_res.boxes) > 0:
            b = max(oa_res.boxes, key=lambda b: b.conf[0])
            oa_cls = int(b.cls)
            oa_prob = float(b.conf)

        # Dibujar caja de rodilla
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 0, 255), 2)

        resultados.append({
            "op": {
                "clase": "normal" if op_cls == 0 else "osteoporosis",
                "prob": op_prob
            },
            "oa": {
                "clase": (
                    "normal-dudoso" if oa_cls in [0, 1]
                    else "leve-moderado" if oa_cls in [2, 3]
                    else "grave"
                ),
                "prob": oa_prob
            }
        })

    return {
        "resultados": resultados,
        "imagenProcesada": to_base64(img),
        "imagenEtiquetada": to_base64(img_draw)
    }

@app.get("/")
def health():
    return {"status": "ok"}
