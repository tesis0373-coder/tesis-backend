import os
import cv2
import numpy as np
import base64
from fastapi import FastAPI, Request
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
# Paths modelos
# ---------------------------
BASE_DIR = os.path.dirname(__file__)
PATH_OP = os.path.join(BASE_DIR, "backend", "det 2cls R2 0.pt")
PATH_OA = os.path.join(BASE_DIR, "backend", "OAyoloIR4AH.pt")

# ---------------------------
# Load models (1 sola vez)
# ---------------------------
@lru_cache(maxsize=1)
def load_models():
    print(">>> Cargando modelos YOLO...")
    model_op = YOLO(PATH_OP)
    model_oa = YOLO(PATH_OA)
    return model_op, model_oa

# ---------------------------
# Utils
# ---------------------------
def to_base64(img):
    _, buffer = cv2.imencode(".jpg", img)
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode()

# ---------------------------
# OP: detectar TODAS las rodillas
# ---------------------------
def detect_all_op(model, img, conf=0.0):
    results = model(img)
    boxes = []

    for r in results:
        for b in r.boxes:
            if float(b.conf[0]) >= conf:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                boxes.append({
                    "cls": int(b.cls[0]),
                    "prob": float(b.conf[0]),
                    "x1": x1, "y1": y1,
                    "x2": x2, "y2": y2
                })
    return boxes

# ---------------------------
# Recorte global (ambas rodillas)
# ---------------------------
def crop_global(img, boxes):
    x1 = min(b["x1"] for b in boxes)
    y1 = min(b["y1"] for b in boxes)
    x2 = max(b["x2"] for b in boxes)
    y2 = max(b["y2"] for b in boxes)
    return img[y1:y2, x1:x2], x1, y1

# ---------------------------
# OA por rodilla
# ---------------------------
def detect_oa_per_knee(model, img, box, conf=0.0):
    crop = img[box["y1"]:box["y2"], box["x1"]:box["x2"]]
    results = model(crop)

    detections = []
    for r in results:
        for b in r.boxes:
            if float(b.conf[0]) >= conf:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                detections.append({
                    "cls": int(b.cls[0]),
                    "prob": float(b.conf[0]),
                    "x1": x1, "y1": y1,
                    "x2": x2, "y2": y2
                })
    return detections

# ---------------------------
# Etiquetado final
# ---------------------------
def label_all(crop, op_boxes, oa_boxes, offx, offy):
    img = crop.copy()

    # OP
    for b in op_boxes:
        x1 = b["x1"] - offx
        y1 = b["y1"] - offy
        x2 = b["x2"] - offx
        y2 = b["y2"] - offy

        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        label = "normal" if b["cls"] == 0 else "osteoporosis"
        cv2.putText(img, label, (x1 + 10, y1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # OA
    for oa in oa_boxes:
        cv2.rectangle(
            img,
            (oa["x1"], oa["y1"]),
            (oa["x2"], oa["y2"]),
            (0, 0, 255),
            2
        )

    return img

# ---------------------------
# API
# ---------------------------
@app.post("/predict")
async def predict(request: Request):
    model_op, model_oa = load_models()

    data = await request.json()
    if "image" not in data:
        return {"detail": "No se recibió imagen"}, 400

    img_b64 = data["image"].split(",")[1]
    img_bytes = base64.b64decode(img_b64)
    img_np = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    if img is None:
        return {"detail": "Imagen inválida"}, 400

    # -------- OP --------
    op_boxes = detect_all_op(model_op, img)
    if len(op_boxes) == 0:
        return {"detail": "No se detectaron rodillas"}, 400

    # -------- Recorte --------
    crop, offx, offy = crop_global(img, op_boxes)

    # -------- OA por rodilla --------
    oa_boxes_global = []
    for b in op_boxes:
        oa_dets = detect_oa_per_knee(model_oa, img, b)
        for oa in oa_dets:
            oa_boxes_global.append({
                "cls": oa["cls"],
                "prob": oa["prob"],
                "x1": b["x1"] + oa["x1"] - offx,
                "y1": b["y1"] + oa["y1"] - offy,
                "x2": b["x1"] + oa["x2"] - offx,
                "y2": b["y1"] + oa["y2"] - offy,
            })

    # -------- Etiquetado --------
    labeled = label_all(crop, op_boxes, oa_boxes_global, offx, offy)

    return {
        "resultado": {
            "rodillas_detectadas": len(op_boxes)
        },
        "imagenProcesada": to_base64(crop),
        "imagenEtiquetada": to_base64(labeled)
    }

# ---------------------------
# Health
# ---------------------------
@app.get("/")
def health():
    return {"status": "ok"}
