import os
import cv2
import numpy as np
import base64
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO

# -------------------------------------------------
# 0) Config FastAPI + CORS
# -------------------------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # luego lo puedes cerrar si quieres
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# 1) Cargar modelos (una sola vez)
# -------------------------------------------------

base_path = os.path.dirname(__file__)

# Rutas absolutas para los modelos
path_corte = os.path.join(base_path, "backend", "corte0.pt")
path_detOP = os.path.join(base_path, "backend", "det 2cls R2 0.pt")
path_detOA = os.path.join(base_path, "backend", "OAyoloIR4AH.pt")


@lru_cache(maxsize=1)
def load_models():
    """
    Carga los modelos YOLO solo la primera vez que se llama.
    Render crea y destruye instancias, pero dentro de la misma
    instancia esto evita recargar en cada request.
    """
    print(">>> [MODELOS] Cargando modelos YOLO...")
    modeldetOP = YOLO(path_detOP)
    modeldetOA = YOLO(path_detOA)
    print(">>> [MODELOS] Modelos cargados OK")
    return modeldetOP, modeldetOA


# -------------------------------------------------
# 2) Funciones de procesamiento (las tuyas)
# -------------------------------------------------

def yolodetOPCrop(modeldet, img, certeza):
    results = modeldet(img)
    cls, prob, crops, coords = [], [], [], []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img[y1:y2, x1:x2]
            conf = box.conf[0].item()

            if conf > certeza:
                cls.append(int(box.cls))
                prob.append(conf)
                crops.append(crop)
                coords.append((x1, y1, x2, y2))

    if not prob:
        # no detectó nada
        return 0, 0, None, 0, 0, 0, 0

    x = prob.index(max(prob))
    return cls[x], prob[x], crops[x], *coords[x]


def yolodetOA(modeldet, img, certeza):
    results = modeldet(img)
    cls, prob, coords = [], [], []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            if conf > certeza:
                cls.append(int(box.cls))
                prob.append(conf)
                coords.append((x1, y1, x2, y2))

    if not prob:
        return 0, 0, 0, 0, 0, 0

    x = prob.index(max(prob))
    return cls[x], prob[x], *coords[x]


def etiquetar(img, clOP, xOP1, yOP1, xOP2, yOP2, clOA, xOA1, yOA1, xOA2, yOA2):
    color = (255, 0, 0)
    grosor = 2

    # OP
    cv2.rectangle(img, (xOP1, yOP1), (xOP2, yOP2), color, grosor)
    etiqueta = "normal" if clOP == 0 else "osteoporosis"
    cv2.putText(
        img,
        etiqueta,
        (xOP1 + 20, yOP1 + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    # OA
    p1 = (xOP1 + xOA1, yOP1 + yOA1)
    p2 = (xOP1 + xOA2, yOP1 + yOA2)
    cv2.rectangle(img, p1, p2, color, grosor)

    if clOA in [0, 1]:
        etiquetaOA = "normal-dudoso"
    elif clOA in [2, 3]:
        etiquetaOA = "leve-moderado"
    else:
        etiquetaOA = "grave"

    cv2.putText(
        img,
        etiquetaOA,
        (p1[0] + 20, p1[1] + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    return img


def to_base64(img):
    _, buffer = cv2.imencode(".jpg", img)
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode()


# -------------------------------------------------
# 3) Esquema del body (para que Swagger MUUESTRE el body)
# -------------------------------------------------

class ImageData(BaseModel):
    image: str


# -------------------------------------------------
# 4) Endpoint principal /predict
# -------------------------------------------------

@app.post("/predict")
async def predict(payload: ImageData):
    print(">>> [API] /predict llamado")

    # Cargar modelos (con cache)
    modeldetOP, modeldetOA = load_models()

    try:
        # 1) Extraer base64 del body
        image_b64 = payload.image

        if "," in image_b64:
            # quitar "data:image/jpeg;base64,"
            image_b64 = image_b64.split(",")[1]

        # 2) Decodificar base64
        try:
            image_bytes = base64.b64decode(image_b64)
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="No se pudo decodificar la imagen en base64",
            )

        nparr = np.frombuffer(image_bytes, np.uint8)
        img_original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_original is None:
            raise HTTPException(
                status_code=400,
                detail="Imagen corrupta o ilegible",
            )

        certeza = 0

        # 3) YOLO Osteoporosis
        print(">>> [API] Ejecutando YOLO OP...")
        clOP, prOP, crop, x1OP, y1OP, x2OP, y2OP = yolodetOPCrop(
            modeldetOP, img_original.copy(), certeza
        )

        if crop is None:
            raise HTTPException(
                status_code=400,
                detail="No se detectó zona de interés (OP)",
            )

        # 4) YOLO Osteoartritis
        print(">>> [API] Ejecutando YOLO OA...")
        clOA, prOA, x1OA, y1OA, x2OA, y2OA = yolodetOA(
            modeldetOA, crop, certeza
        )

        # 5) Etiquetado
        h, w = crop.shape[:2]
        crop_etiquetado = etiquetar(
            crop.copy(), clOP, 0, 0, w - 1, h - 1, clOA, x1OA, y1OA, x2OA, y2OA
        )

        print(">>> [API] Respuesta OK")

        return {
            "resultado": {
                "clase_op": "normal" if clOP == 0 else "osteoporosis",
                "prob_op": float(prOP),
                "clase_oa": (
                    "normal-dudoso"
                    if clOA in [0, 1]
                    else "leve-moderado"
                    if clOA in [2, 3]
                    else "grave"
                ),
                "prob_oa": float(prOA),
            },
            "imagenOriginal": to_base64(img_original),
            "imagenProcesada": to_base64(crop),
            "imagenEtiquetada": to_base64(crop_etiquetado),
        }

    except HTTPException:
        # errores controlados -> se re-lanzan tal cual
        raise
    except Exception as e:
        # cualquier otra cosa -> 500
        print(">>> [API] ERROR no controlado:", e)
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------
# 5) Endpoint de salud
# -------------------------------------------------

@app.get("/")
def health():
    return {"status": "ok", "message": "Backend FastAPI funcionando"}
