from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import os

# ===============================
# PATHS
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")

# ===============================
# CARGA DE MODELOS
# ===============================
modelrecorte = YOLO(os.path.join(BACKEND_DIR, "recorte2.pt"))
modeldetOP  = YOLO(os.path.join(BACKEND_DIR, "3clsOPfft.pt"))
modeldetOA  = YOLO(os.path.join(BACKEND_DIR, "OAyoloR4cls5.pt"))

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
# FUNCIONES IA
# ===============================
def yolorecorte(model, img):
    results = model(img)
    coords = []
    for r in results:
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            coords.append([x1, y1, x2, y2])
    return coords


def procesar_fft(crop):
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    f = np.fft.fft2(crop)
    fshift = np.fft.fftshift(f)
    ms = 20 * np.log(np.abs(fshift) + 1)
    ms = np.clip(ms, 0, 255).astype(np.uint8)

    return ms


def detectar_op(model, fft_img):
    r = model(fft_img)[0]
    cls = int(r.probs.top1)
    prob = float(r.probs.top1conf)
    return cls, prob


def detectar_oa(model, fft_img):
    r = model(fft_img)[0]
    box = r.boxes[0]
    return int(box.cls), float(box.conf), *map(int, box.xyxy[0])


def etiquetar(img, cl_op, cl_oa, x1, y1, x2, y2):
    etiquetas_op = ["Normal", "Osteopenia", "Osteoporosis"]
    etiquetas_oa = ["OA dudoso", "OA leve", "OA moderado", "OA grave"]

    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(
        img,
        etiquetas_oa[cl_oa],
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    return img

# ===============================
# PIPELINE PRINCIPAL
# ===============================
def correr_modelo(img):
    coords = yolorecorte(modelrecorte, img)

    if not coords:
        raise ValueError("No se detectó región de interés")

    x1, y1, x2, y2 = coords[0]
    crop = img[y1:y2, x1:x2]

    fft_img = procesar_fft(crop)

    cl_op, prob_op = detectar_op(modeldetOP, fft_img)
    cl_oa, prob_oa, xa1, ya1, xa2, ya2 = detectar_oa(modeldetOA, fft_img)

    # Imagen PROCESADA (solo FFT)
    img_procesada = fft_img.copy()

    # Imagen ETIQUETADA (FFT + cajas)
    img_etiquetada = cv2.cvtColor(fft_img, cv2.COLOR_GRAY2BGR)
    img_etiquetada = etiquetar(img_etiquetada, cl_op, cl_oa, xa1, ya1, xa2, ya2)

    return img_procesada, img_etiquetada, cl_op, prob_op, cl_oa, prob_oa

# ===============================
# ENDPOINT
# ===============================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Imagen inválida")

        img_proc, img_etq, cl_op, prob_op, cl_oa, prob_oa = correr_modelo(img)

        _, buf_p = cv2.imencode(".jpg", img_proc)
        _, buf_e = cv2.imencode(".jpg", img_etq)

        return JSONResponse({
            "mensaje": "Análisis completado correctamente",
            "resultado": {
                "clase_op": cl_op,
                "prob_op": prob_op,
                "clase_oa": cl_oa,
                "prob_oa": prob_oa
            },
            "imagenProcesada": f"data:image/jpeg;base64,{base64.b64encode(buf_p).decode()}",
            "imagenEtiquetada": f"data:image/jpeg;base64,{base64.b64encode(buf_e).decode()}"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
