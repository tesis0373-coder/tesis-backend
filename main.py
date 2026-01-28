from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import os

# ===============================
# PATH BASE
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")

# ===============================
# 1) CARGA DE MODELOS
# ===============================
modelrecorte = YOLO(os.path.join(BACKEND_DIR, "recorte2.pt"))
modeldetOP  = YOLO(os.path.join(BACKEND_DIR, "3clsOPfft.pt"))
modeldetOA  = YOLO(os.path.join(BACKEND_DIR, "OAyoloR4cls5.pt"))

# ===============================
# 2) FUNCIONES IA
# ===============================

def yolorecorte(model, img):
    results = model(img)
    coords = []
    for r in results:
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            coords.append([x1, y1, x2, y2])
    return coords


def yolodetOPCrop(model, crop):
    """
    FFT SOLO PARA EL MODELO (NO SE DEVUELVE)
    """
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    f = np.fft.fft2(crop)
    fshift = np.fft.fftshift(f)
    ms = 20 * np.log(np.abs(fshift) + 1)
    ms = ms.astype(np.uint8)

    r = model(ms)[0]
    return int(r.probs.top1), float(r.probs.top1conf)


def yolodetOA(model, crop, certeza=0.0):
    results = model(crop)
    for r in results:
        for b in r.boxes:
            if b.conf[0].item() > certeza:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                return int(b.cls), float(b.conf[0]), x1, y1, x2, y2
    return 0, 0.0, 0, 0, 0, 0


def CorrerModelo(img):
    coords = yolorecorte(modelrecorte, img)

    img_procesada = None
    img_etiquetada = None

    for c in coords:
        # ---------------------------
        # RECORTE ESPACIAL
        # ---------------------------
        crop = img[c[1]:c[3], c[0]:c[2]]

        # ---------------------------
        # IMAGEN PROCESADA (VISIBLE)
        # ---------------------------
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        img_procesada = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2BGR)

        # ---------------------------
        # MODELOS
        # ---------------------------
        clOP, _ = yolodetOPCrop(modeldetOP, crop)
        clOA, _, xa1, ya1, xa2, ya2 = yolodetOA(modeldetOA, crop)

        # ---------------------------
        # IMAGEN ETIQUETADA
        # ---------------------------
        img_etiquetada = img_procesada.copy()

        etiquetas_op = ["Sin osteoporosis", "Osteopenia", "Osteoporosis"]
        cv2.putText(
            img_etiquetada,
            etiquetas_op[clOP],
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        etiquetas_oa = ["Sin OA", "OA dudoso", "OA leve", "OA moderado", "OA grave"]
        cv2.rectangle(
            img_etiquetada,
            (xa1, ya1),
            (xa2, ya2),
            (255, 0, 0),
            2
        )
        cv2.putText(
            img_etiquetada,
            etiquetas_oa[clOA],
            (xa1, max(ya1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        break  # solo un ROI

    return img_procesada, img_etiquetada

# ===============================
# 3) FASTAPI
# ===============================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse({"error": "Imagen inválida"}, status_code=400)

    img_proc, img_label = CorrerModelo(img)

    _, buf_proc = cv2.imencode(".jpg", img_proc)
    _, buf_label = cv2.imencode(".jpg", img_label)

    return JSONResponse({
        "imagen_procesada": f"data:image/jpeg;base64,{base64.b64encode(buf_proc).decode()}",
        "imagen_etiquetada": f"data:image/jpeg;base64,{base64.b64encode(buf_label).decode()}"
    })
