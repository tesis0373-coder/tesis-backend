import os
import io
import cv2
import numpy as np
from functools import lru_cache
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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
# RUTAS DE MODELOS
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")

PATH_RECORTE = os.path.join(BACKEND_DIR, "recorte2.pt")
PATH_OP = os.path.join(BACKEND_DIR, "3clsOPfft.pt")
PATH_OA = os.path.join(BACKEND_DIR, "OAyoloR4cls5.pt")

# ---------------------------
# CARGA DE MODELOS (como en el main que funcionaba)
# ---------------------------
@lru_cache(maxsize=1)
def load_models():
    if not os.path.exists(PATH_RECORTE):
        raise RuntimeError(f"No existe {PATH_RECORTE}")
    if not os.path.exists(PATH_OP):
        raise RuntimeError(f"No existe {PATH_OP}")
    if not os.path.exists(PATH_OA):
        raise RuntimeError(f"No existe {PATH_OA}")

    return (
        YOLO(PATH_RECORTE),
        YOLO(PATH_OP),
        YOLO(PATH_OA)
    )

# ---------------------------
# FUNCIONES
# ---------------------------
def yolorecorte(model, img):
    results = model(img)
    coor = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            coor.append([x1, y1, x2, y2])
    return coor

def yolodetOPCrop(model, crop):
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    f = np.fft.fftshift(np.fft.fft2(crop))
    ms = (20 * np.log(np.abs(f) + 1)).astype(np.uint8)

    results = model(ms)
    cls = int(results[0].probs.top1)
    prob = float(results[0].probs.top1conf)
    return cls, prob

def yolodetOA(model, crop, certeza=0.0):
    results = model(crop)
    best = None
    best_conf = -1

    for r in results:
        for box in r.boxes:
            conf = box.conf[0].item()
            if conf > certeza and conf > best_conf:
                best = box
                best_conf = conf

    if best is None:
        return None

    x1, y1, x2, y2 = map(int, best.xyxy[0])
    return int(best.cls[0]), float(best.conf[0]), x1, y1, x2, y2

def etiquetar2(imagen, clOP, xOP1, yOP1, xOP2, yOP2, clOA, xOA1, yOA1, xOA2, yOA2):
    cv2.rectangle(imagen, (xOP1, yOP1), (xOP2, yOP2), (255, 0, 0), 2)

    etiqueta_op = ["Sin osteoporosis", "Osteopenia", "Osteoporosis"][clOP]
    cv2.putText(imagen, etiqueta_op, (xOP1, yOP1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.rectangle(
        imagen,
        (xOP1 + xOA1, yOP1 + yOA1),
        (xOP1 + xOA2, yOP1 + yOA2),
        (0, 0, 255), 2
    )

    etiquetas_oa = [
        "Sin OA", "OA dudoso", "OA leve", "OA moderado", "OA grave"
    ]
    cv2.putText(
        imagen,
        etiquetas_oa[clOA],
        (xOP1 + xOA1, yOP1 + yOA1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )

    return imagen

def CorrerModelo(img):
    modelrecorte, modelOP, modelOA = load_models()
    certeza = 0.0

    coor = yolorecorte(modelrecorte, img)
    if not coor:
        raise HTTPException(status_code=400, detail="No se detectó rodilla")

    for c in coor:
        crop = img[c[1]:c[3], c[0]:c[2]]
        clOP, _ = yolodetOPCrop(modelOP, crop)
        oa = yolodetOA(modelOA, crop, certeza)

        if oa:
            clOA, _, x1, y1, x2, y2 = oa
            img = etiquetar2(img, clOP, c[0], c[1], c[2], c[3], clOA, x1, y1, x2, y2)

    return img

# ---------------------------
# API
# ---------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Imagen inválida")

    result = CorrerModelo(img)
    _, buffer = cv2.imencode(".jpg", result)

    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")

@app.get("/")
def health():
    return {"status": "ok"}
