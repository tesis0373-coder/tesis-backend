from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import base64
import os
from ultralytics import YOLO

# ===============================
# FASTAPI
# ===============================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# SCHEMA
# ===============================
class PredictRequest(BaseModel):
    image: str  # base64 limpio

# ===============================
# PATHS
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "backend")

# ===============================
# MODELOS
# ===============================
modelrecorte = YOLO(os.path.join(MODELS_DIR, "recorte2.pt"))
modeldetOP   = YOLO(os.path.join(MODELS_DIR, "3clsOPfft.pt"))
modeldetOA   = YOLO(os.path.join(MODELS_DIR, "OAyoloR4cls5.pt"))

# ===============================
# FUNCIONES
# ===============================

def normalizar_imagen(img):
    """
    Fuerza imagen a formato estable:
    - BGR
    - 8 bits
    - sin EXIF
    """
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (1024, 1024))
    return img


def yolorecorte(model, img):
    results = model(img)
    cajas = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cajas.append([x1, y1, x2, y2])
    return cajas


def filtrar_rodillas(cajas, ancho_img):
    """
    Devuelve máximo 2 rodillas (izquierda y derecha),
    seleccionando la de mayor área por lado.
    """
    if len(cajas) <= 2:
        return cajas

    centro = ancho_img // 2
    izquierda, derecha = [], []

    for x1, y1, x2, y2 in cajas:
        cx = (x1 + x2) // 2
        area = (x2 - x1) * (y2 - y1)
        if cx < centro:
            izquierda.append((area, [x1, y1, x2, y2]))
        else:
            derecha.append((area, [x1, y1, x2, y2]))

    rodillas = []
    if izquierda:
        rodillas.append(max(izquierda, key=lambda x: x[0])[1])
    if derecha:
        rodillas.append(max(derecha, key=lambda x: x[0])[1])

    return rodillas

#esto se comenta para OP
# def yolodetOPCrop(model, crop):
#       if crop.ndim == 3:
#           crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

#       f = np.fft.fft2(crop)
#       fshift = np.fft.fftshift(f)
#       ms = 20 * np.log(np.abs(fshift) + 1)
#       ms = ms.astype(np.uint8)

#       results = model(ms)
#       for r in results:
#           cls = int(r.probs.top1)
#           prob = float(r.probs.top1conf)
#           return cls, prob

#       return 0, 0.0


def yolodetOA(model, crop, certeza=0):
    results = model(crop)
    best = None
    best_prob = 0

    for r in results:
        for box in r.boxes:
            conf = box.conf[0].item()
            if conf > certeza and conf > best_prob:
                best_prob = conf
                best = (
                    int(box.cls),
                    conf,
                    *map(int, box.xyxy[0])
                )
    return best

#clOP
def etiquetar2(img, x1, y1, x2, y2, clOA=None, boxOA=None):
    # # ---- Rodilla ---- esto se comenta para OP
    # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # #  # ---- OP (lógica clínica original) ----
    # if clOP == 0:
    #       texto_op = "Sin osteoporosis"
    # elif clOP == 1:
    #       texto_op = "Osteopenia"
    # else:
    #       texto_op = "Osteoporosis"

    # cv2.putText(
    #       img,
    #       f"OP: {texto_op}",
    #       (x1, y1 - 10),
    #       cv2.FONT_HERSHEY_SIMPLEX,
    #       0.7,
    #       (0, 255, 0),
    #       2
    #   )

    # ---- OA ----
    if clOA is not None and boxOA is not None:
        xa1, ya1, xa2, ya2 = boxOA
        cv2.rectangle(
            img,
            (x1 + xa1, y1 + ya1),
            (x1 + xa2, y1 + ya2),
            (0, 0, 255),
            2
        )

        if clOA == 3:
            texto_oa = "Sin osteoartrosis"
        elif clOA == 0:
            texto_oa = "OA dudoso"
        elif clOA == 4:
            texto_oa = "OA leve"
        elif clOA == 1:
            texto_oa = "OA moderado"
        else:
            texto_oa = "OA grave"

        cv2.putText(
            img,
            f"OA: {texto_oa}",
            (x1 + xa1, y1 + ya1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

    return img

# ===============================
# ENDPOINT
# ===============================
@app.post("/predict")
def predict(data: PredictRequest):
    try:
        # ---- Decode ----
        img_bytes = base64.b64decode(data.image)
        np_img = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Imagen inválida")

        img = normalizar_imagen(img)
        img_etiquetada = img.copy()

        # ---- Detectar rodillas ----
        rodillas_raw = yolorecorte(modelrecorte, img)
        if len(rodillas_raw) == 0:
            raise ValueError("No se detectaron rodillas")

        rodillas = filtrar_rodillas(rodillas_raw, img.shape[1])

        # ---- Imagen procesada ----
        x1 = min(r[0] for r in rodillas)
        y1 = min(r[1] for r in rodillas)
        x2 = max(r[2] for r in rodillas)
        y2 = max(r[3] for r in rodillas)
        imagen_procesada = img[y1:y2, x1:x2].copy()

        resultados = []

        # ---- Analizar cada rodilla ----
        for rx1, ry1, rx2, ry2 in rodillas:
            crop = img[ry1:ry2, rx1:rx2].copy()
            h, w = crop.shape[:2]
            if h < 50 or w < 50:
                continue
            clOP, probOP = yolodetOPCrop(modeldetOP, crop)
            oa = yolodetOA(modeldetOA, crop)

            if oa:
                clOA, probOA, xa1, ya1, xa2, ya2 = oa
                boxOA = (xa1, ya1, xa2, ya2)
            else:
                clOA = probOA = boxOA = None

            img_etiquetada = etiquetar2(
                img_etiquetada,
                rx1, ry1, rx2, ry2,
                #clOP,#esto se comenta
                clOA, 
                boxOA
            )

            resultados.append({
                #esto se comenta
                #"clase_op": clOP,
                #"prob_op": probOP,
                "clase_oa": clOA,
                "prob_oa": probOA
            })

        # ---- Encode ----
        _, buf_proc = cv2.imencode(".jpg", imagen_procesada)
        _, buf_et = cv2.imencode(".jpg", img_etiquetada)

        return {
            "resultado": resultados[0],
            "imagenProcesada": "data:image/jpeg;base64," + base64.b64encode(buf_proc).decode(),
            "imagenEtiquetada": "data:image/jpeg;base64," + base64.b64encode(buf_et).decode()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
