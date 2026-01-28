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

RECORTE_PATH = os.path.join(MODEL_DIR, "recorte2.pt")
OP_PATH      = os.path.join(MODEL_DIR, "3clsOPfft.pt")
OA_PATH      = os.path.join(MODEL_DIR, "OAyoloR4cls5.pt")

# ===============================
# CARGA DE MODELOS
# ===============================
modelrecorte = YOLO(RECORTE_PATH)
modeldetOP   = YOLO(OP_PATH)     # clasificador por FFT (3 clases)
modeldetOA   = YOLO(OA_PATH)     # detector OA (cajas en la rodilla recortada)

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

@app.get("/health")
def health():
    return {"ok": True}

# ===============================
# FUNCIONES
# ===============================
def yolorecorte_primera_caja(model, img):
    """
    Devuelve la PRIMERA caja [x1,y1,x2,y2] o None si no detecta.
    """
    results = model(img)
    for r in results:
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            # clamp por si acaso
            h, w = img.shape[:2]
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h))
            if x2 > x1 and y2 > y1:
                return [x1, y1, x2, y2]
    return None


def op_fft_classifier(model, crop_bgr):
    """
    Usa FFT sobre GRAY para clasificar OP (normal/osteopenia/osteoporosis).
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = 20 * np.log(np.abs(fshift) + 1.0)
    mag = mag.astype(np.uint8)

    r = model(mag)[0]
    cls = int(r.probs.top1)
    conf = float(r.probs.top1conf)
    return cls, conf


def oa_detector_primera_caja(model, crop_bgr, certeza=0.0):
    """
    Detecta OA en el crop (coordenadas RELATIVAS al crop).
    Devuelve: (cls, conf, x1,y1,x2,y2) o (0,0,0,0,0,0) si nada.
    """
    results = model(crop_bgr)
    best = None

    for r in results:
        for b in r.boxes:
            conf = float(b.conf[0].item())
            if conf >= certeza:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                cls = int(b.cls[0].item())
                if best is None or conf > best[1]:
                    best = (cls, conf, x1, y1, x2, y2)

    if best is None:
        return 0, 0.0, 0, 0, 0, 0
    return best


def draw_labels(img_bgr, clOP, clOA, oa_box):
    """
    Dibuja sobre LA IMAGEN RECORTADA (procesada) los textos y el cuadro de OA.
    """
    etiquetas_op = ["Normal", "Osteopenia", "Osteoporosis"]
    etiquetas_oa = ["Normal", "OA dudoso", "OA leve", "OA moderado", "OA grave"]

    # Texto OP (arriba a la izquierda)
    cv2.putText(
        img_bgr,
        f"OP: {etiquetas_op[clOP]}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    # Texto OA (debajo)
    cv2.putText(
        img_bgr,
        f"OA: {etiquetas_oa[clOA]}",
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    # Caja OA
    x1, y1, x2, y2 = oa_box
    if x2 > x1 and y2 > y1:
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return img_bgr


def encode_b64_jpg(img_bgr):
    ok, buffer = cv2.imencode(".jpg", img_bgr)
    if not ok:
        raise ValueError("No se pudo codificar la imagen")
    return base64.b64encode(buffer).decode("utf-8")


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
            raise HTTPException(status_code=400, detail="Imagen inválida")

        # 1) Recorte
        box = yolorecorte_primera_caja(modelrecorte, img)
        if box is None:
            # si no recorta, usamos toda la imagen (pero sin romper)
            crop = img.copy()
        else:
            x1, y1, x2, y2 = box
            crop = img[y1:y2, x1:x2].copy()

        # 2) OP (clasificación por FFT) -> SOLO para resultado, no para imagen
        clOP, probOP = op_fft_classifier(modeldetOP, crop)

        # 3) OA (detección) en el crop
        clOA, probOA, xa1, ya1, xa2, ya2 = oa_detector_primera_caja(modeldetOA, crop, certeza=0.0)

        # 4) Imagen procesada (SIN recuadros): SOLO el crop
        img_procesada = crop.copy()

        # 5) Imagen etiquetada: crop + caja OA + textos
        img_etiquetada = crop.copy()
        img_etiquetada = draw_labels(img_etiquetada, clOP, clOA, (xa1, ya1, xa2, ya2))

        # 6) Encode base64
        b64_proc = encode_b64_jpg(img_procesada)
        b64_etiq = encode_b64_jpg(img_etiquetada)

        # 7) Respuesta JSON (para que el front NO truene)
        return JSONResponse({
            "resultado": {
                "clase_op": ["normal", "osteopenia", "osteoporosis"][clOP],
                "prob_op": probOP,
                "clase_oa": ["normal", "oa-dudoso", "oa-leve", "oa-moderado", "oa-grave"][clOA],
                "prob_oa": probOA
            },
            "imagen_procesada": f"data:image/jpeg;base64,{b64_proc}",
            "imagen_etiquetada": f"data:image/jpeg;base64,{b64_etiq}"
        })

    except HTTPException:
        raise
    except Exception as e:
        # Esto también es JSON para no romper el front
        return JSONResponse({"error": str(e)}, status_code=500)
