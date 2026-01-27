from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO
import io
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ---------------------------
# 1) CARGA DE MODELOS
# ---------------------------
modelrecorte = YOLO(os.path.join(BASE_DIR, "backend", "recorte2.pt"))
modeldetOP = YOLO(os.path.join(BASE_DIR, "backend", "3clsOPfft.pt"))
modeldetOA = YOLO(os.path.join(BASE_DIR, "backend", "OAyoloR4cls5.pt"))

# ---------------------------
# 2) FUNCIONES
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

    f = np.fft.fft2(crop)
    fshift = np.fft.fftshift(f)
    ms = 20 * np.log(np.abs(fshift) + 1)
    ms = ms.astype(np.uint8)

    results = model(ms)
    for r in results:
        cls = int(r.probs.top1)
        prob = float(r.probs.top1conf)
    return cls, prob


def yolodetOA(model, crop, certeza=0):
    results = model(crop)
    cls, prob, coords = [], [], []

    for r in results:
        for box in r.boxes:
            conf = box.conf[0].item()
            if conf > certeza:
                cls.append(int(box.cls))
                prob.append(conf)
                coords.append(tuple(map(int, box.xyxy[0])))

    i = prob.index(max(prob))
    return cls[i], prob[i], *coords[i]


def etiquetar2(img, clOP, x1, y1, x2, y2, clOA, xa1, ya1, xa2, ya2):
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    txt_op = ["Sin osteoporosis", "Osteopenia", "Osteoporosis"][clOP]
    cv2.putText(img, txt_op, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.rectangle(
        img,
        (x1 + xa1, y1 + ya1),
        (x1 + xa2, y1 + ya2),
        (0, 0, 255),
        2
    )

    oa_txt = ["Sin OA", "OA dudoso", "OA leve", "OA moderado", "OA grave"][clOA]
    cv2.putText(
        img, oa_txt,
        (x1 + xa1, y1 + ya1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
    )

    return img


def correr_modelo(img):
    coor = yolorecorte(modelrecorte, img)

    for c in coor:
        crop = img[c[1]:c[3], c[0]:c[2]]
        clOP, _ = yolodetOPCrop(modeldetOP, crop)
        clOA, _, xa1, ya1, xa2, ya2 = yolodetOA(modeldetOA, crop)
        img = etiquetar2(img, clOP, *c, clOA, xa1, ya1, xa2, ya2)

    return img

# ---------------------------
# 3) FASTAPI
# ---------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict", response_class=StreamingResponse)
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    result = correr_modelo(img)
    _, encoded = cv2.imencode(".jpg", result)

    return StreamingResponse(
        io.BytesIO(encoded.tobytes()),
        media_type="image/jpeg"
    )
