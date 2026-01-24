import os
import cv2
import base64
import numpy as np
from functools import lru_cache
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

# ===========================
# APP
# ===========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================
# MODELOS
# ===========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")

PATH_RECORTE = os.path.join(BACKEND_DIR, "recorte2.pt")
PATH_OP = os.path.join(BACKEND_DIR, "3clsOPfft.pt")
PATH_OA = os.path.join(BACKEND_DIR, "OAyoloR4cls5.pt")

@lru_cache(maxsize=1)
def load_models():
    return YOLO(PATH_RECORTE), YOLO(PATH_OP), YOLO(PATH_OA)

# ===========================
# UTILIDADES
# ===========================
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

# ===========================
# DETECCIONES
# ===========================
def detectar_rodillas(model, img):
    results = model(img)
    boxes = []
    for r in results:
        for b in r.boxes:
            boxes.append(list(map(int, b.xyxy[0])))
    return boxes

def detectar_OP(model, crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    f = np.fft.fftshift(np.fft.fft2(gray))
    ms = (20 * np.log(np.abs(f) + 1)).astype(np.uint8)
    res = model(ms)
    return int(res[0].probs.top1), float(res[0].probs.top1conf)

def detectar_OA(model, crop):
    res = model(crop)
    best = None
    best_conf = -1
    for r in res:
        for b in r.boxes:
            if b.conf[0] > best_conf:
                best_conf = b.conf[0]
                best = b
    if best is None:
        return None
    x1, y1, x2, y2 = map(int, best.xyxy[0])
    return int(best.cls[0]), float(best.conf[0]), x1, y1, x2, y2

# ===========================
# PIPELINE PRINCIPAL
# ===========================
def correr_modelo(img):
    model_rec, model_op, model_oa = load_models()

    boxes = detectar_rodillas(model_rec, img)
    if not boxes:
        raise HTTPException(status_code=400, detail="No se detectaron rodillas")

    # --------
    # 1 o 2 rodillas → caja envolvente
    # --------
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)

    crop = img[y1:y2, x1:x2].copy()

    # Imagen procesada (base visual)
    procesada = cv2.normalize(crop, None, 0, 255, cv2.NORM_MINMAX)

    # Imagen etiquetada
    etiquetada = procesada.copy()

    # Resultados
    clase_op = "normal"
    prob_op = 0.0
    clase_oa = "normal"
    prob_oa = 0.0

    # --------
    # Analizar cada rodilla por separado
    # --------
    for b in boxes:
        bx1, by1, bx2, by2 = b

        # Coordenadas relativas al crop
        rx1, ry1 = bx1 - x1, by1 - y1
        rx2, ry2 = bx2 - x1, by2 - y1

        rodilla = procesada[ry1:ry2, rx1:rx2]

        # OP
        cl_op, p_op = detectar_OP(model_op, rodilla)
        clase_op = ["normal", "osteopenia", "osteoporosis"][cl_op]
        prob_op = max(prob_op, p_op)

        cv2.rectangle(etiquetada, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)
        cv2.putText(
            etiquetada,
            clase_op,
            (rx1, max(30, ry1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # OA
        oa = detectar_OA(model_oa, rodilla)
        if oa:
            cl, p, ox1, oy1, ox2, oy2 = oa
            clase_oa = ["normal", "dudoso", "leve", "moderado", "grave"][cl]
            prob_oa = max(prob_oa, p)
            cv2.rectangle(
                etiquetada,
                (rx1 + ox1, ry1 + oy1),
                (rx1 + ox2, ry1 + oy2),
                (0, 0, 255),
                2
            )

    return procesada, etiquetada, clase_op, prob_op, clase_oa, prob_oa

# ===========================
# API
# ===========================
@app.post("/predict")
async def predict(req: Request):
    data = await req.json()
    img = decode_base64_image(data["image"])

    proc, etq, cl_op, p_op, cl_oa, p_oa = correr_modelo(img)

    return {
        "resultado": {
            "clase_op": cl_op,
            "prob_op": p_op,
            "clase_oa": cl_oa,
            "prob_oa": p_oa,
        },
        "imagenProcesada": to_base64(proc),
        "imagenEtiquetada": to_base64(etq),
    }

@app.get("/")
def health():
    return {"status": "ok"}
