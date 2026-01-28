from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
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

# ===============================
# CARGA DE MODELOS
# ===============================
modelrecorte = YOLO(os.path.join(MODEL_DIR, "recorte2.pt"))
modeldetOP = YOLO(os.path.join(MODEL_DIR, "3clsOPfft.pt"))
modeldetOA = YOLO(os.path.join(MODEL_DIR, "OAyoloR4cls5.pt"))

# ===============================
# FUNCIONES
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
    mag = 20 * np.log(np.abs(fshift) + 1)
    return mag.astype(np.uint8)


def yolodetOP(model, crop):
    fft_img = procesar_fft(crop)
    r = model(fft_img)[0]
    return int(r.probs.top1), float(r.probs.top1conf)


def yolodetOA(model, crop):
    r = model(crop)[0]
    if len(r.boxes) == 0:
        return 0, 0.0, 0, 0, 0, 0

    b = r.boxes[0]
    x1, y1, x2, y2 = map(int, b.xyxy[0])
    return int(b.cls), float(b.conf), x1, y1, x2, y2


def etiquetar(img, c, clOP, clOA, oa_box):
    x1, y1, x2, y2 = c
    xa1, ya1, xa2, ya2 = oa_box

    etiquetas_op = ["Normal", "Osteopenia", "Osteoporosis"]
    etiquetas_oa = ["Normal", "OA dudoso", "OA leve", "OA moderado", "OA grave"]

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(img, etiquetas_op[clOP], (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if xa2 > xa1 and ya2 > ya1:
        cv2.rectangle(
            img,
            (x1 + xa1, y1 + ya1),
            (x1 + xa2, y1 + ya2),
            (255, 0, 0),
            2
        )
        cv2.putText(img, etiquetas_oa[clOA],
                    (x1 + xa1, y1 + ya1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return img


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
# ENDPOINT
# ===============================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return PlainTextResponse("Error: imagen inválida", status_code=400)

        coords = yolorecorte(modelrecorte, img)

        # ---------- IMAGEN PROCESADA (SIN CUADROS) ----------
        img_procesada = None

        if coords:
            c = coords[0]
            crop = img[c[1]:c[3], c[0]:c[2]]
            img_procesada = procesar_fft(crop)
            img_procesada = cv2.cvtColor(img_procesada, cv2.COLOR_GRAY2BGR)
        else:
            img_procesada = img.copy()

        # ---------- IMAGEN ETIQUETADA ----------
        img_etiquetada = img_procesada.copy()

        for c in coords:
            crop = img[c[1]:c[3], c[0]:c[2]]
            clOP, _ = yolodetOP(modeldetOP, crop)
            clOA, _, xa1, ya1, xa2, ya2 = yolodetOA(modeldetOA, crop)

            img_etiquetada = etiquetar(
                img_etiquetada,
                [0, 0, img_procesada.shape[1], img_procesada.shape[0]],
                clOP,
                clOA,
                [xa1, ya1, xa2, ya2]
            )

        # ---------- CODIFICACIÓN ----------
        _, buf1 = cv2.imencode(".jpg", img_procesada)
        _, buf2 = cv2.imencode(".jpg", img_etiquetada)

        img1_b64 = base64.b64encode(buf1).decode()
        img2_b64 = base64.b64encode(buf2).decode()

        # 🔥 FRONTEND SOLO QUIERE TEXTO → NO JSON
        return PlainTextResponse("Análisis completado correctamente")

    except Exception as e:
        return PlainTextResponse(f"Error: {str(e)}", status_code=500)
