from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import base64
from ultralytics import YOLO
import os

# ===============================
# PATHS
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "backend")

modelrecorte = YOLO(os.path.join(MODEL_DIR, "recorte2.pt"))
modeldetOP   = YOLO(os.path.join(MODEL_DIR, "3clsOPfft.pt"))
modeldetOA   = YOLO(os.path.join(MODEL_DIR, "OAyoloR4cls5.pt"))

# ===============================
# FASTAPI
# ===============================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# FUNCIONES
# ===============================
def encode_b64(img):
    ok, buffer = cv2.imencode(".jpg", img)
    if not ok:
        raise ValueError("No se pudo codificar imagen")
    return base64.b64encode(buffer).decode("utf-8")


def recorte_rodilla(img):
    results = modelrecorte(img)
    for r in results:
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            return img[y1:y2, x1:x2].copy()
    return img.copy()  # fallback


def clasificar_op(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    f = np.fft.fftshift(np.fft.fft2(gray))
    mag = (20 * np.log(np.abs(f) + 1)).astype(np.uint8)

    r = modeldetOP(mag)[0]
    return int(r.probs.top1), float(r.probs.top1conf)


def detectar_oa(crop):
    results = modeldetOA(crop)
    for r in results:
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            return int(b.cls[0]), float(b.conf[0]), x1, y1, x2, y2
    return 0, 0.0, None


def etiquetar(img, clOP, clOA, box):
    etiquetas_op = ["Normal", "Osteopenia", "Osteoporosis"]
    etiquetas_oa = ["Normal", "OA dudoso", "OA leve", "OA moderado", "OA grave"]

    cv2.putText(img, f"OP: {etiquetas_op[clOP]}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.putText(img, f"OA: {etiquetas_oa[clOA]}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    if box:
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 2)

    return img

# ===============================
# ENDPOINT
# ===============================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Imagen inválida")

        # 1) Recorte
        crop = recorte_rodilla(img)

        # 2) Clasificaciones
        clOP, probOP = clasificar_op(crop)
        clOA, probOA, box = detectar_oa(crop)

        # 3) Imágenes
        imagenProcesada  = crop.copy()
        imagenEtiquetada = etiquetar(crop.copy(), clOP, clOA, box)

        return JSONResponse({
            "resultado": {
                "clase_op": ["normal", "osteopenia", "osteoporosis"][clOP],
                "prob_op": probOP,
                "clase_oa": ["normal", "oa-dudoso", "oa-leve", "oa-moderado", "oa-grave"][clOA],
                "prob_oa": probOA
            },
            "imagenProcesada":  f"data:image/jpeg;base64,{encode_b64(imagenProcesada)}",
            "imagenEtiquetada": f"data:image/jpeg;base64,{encode_b64(imagenEtiquetada)}"
        })

    except Exception as e:
        return JSONResponse({"detail": str(e)}, status_code=500)
