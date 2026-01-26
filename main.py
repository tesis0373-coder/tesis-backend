from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO
import base64

# ---------------------------
# 1) CARGA DE MODELOS
# ---------------------------

modelrecorte = YOLO("backend/recorte2.pt")
modeldetOP = YOLO("backend/3clsOPfft.pt")
modeldetOA = YOLO("backend/OAyoloR4cls5.pt")

# ---------------------------
# 2) FUNCIONES ORIGINALES
# ---------------------------

def yolorecorte(model, img):
    results = model(img)
    coor = []
    for r in results:
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            coor.append([x1, y1, x2, y2])
    return coor


def yolodetOPCrop(model, crop):
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    f = np.fft.fft2(crop)
    fshift = np.fft.fftshift(f)
    ms = 20 * np.log(np.abs(fshift) + 1)
    ms = ms.astype(np.uint8)

    r = model(ms)[0]
    cls = int(r.probs.top1)
    prob = float(r.probs.top1conf)
    return cls, prob


def yolodetOA(model, img, certeza):
    results = model(img)
    cls, prob, box = None, None, None

    for r in results:
        for b in r.boxes:
            conf = float(b.conf[0])
            if conf > certeza:
                cls = int(b.cls)
                prob = conf
                box = list(map(int, b.xyxy[0]))

    return cls, prob, box


def etiquetar2(img, clOP, boxOP, clOA, boxOA):
    x1, y1, x2, y2 = boxOP
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    labelOP = ["Sin osteoporosis", "Osteopenia", "Osteoporosis"][clOP]
    cv2.putText(img, labelOP, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    if boxOA:
        xa1, ya1, xa2, ya2 = boxOA
        cv2.rectangle(
            img,
            (x1 + xa1, y1 + ya1),
            (x1 + xa2, y1 + ya2),
            (0, 0, 255),
            2
        )
        labelsOA = [
            "Sin OA", "OA dudoso", "OA leve", "OA moderado", "OA grave"
        ]
        cv2.putText(
            img,
            labelsOA[clOA],
            (x1 + xa1, y1 + ya1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )
    return img


def CorrerModelo(img):
    coor = yolorecorte(modelrecorte, img)

    for c in coor:
        crop = img[c[1]:c[3], c[0]:c[2]]
        clOP, _ = yolodetOPCrop(modeldetOP, crop)
        clOA, _, boxOA = yolodetOA(modeldetOA, crop, 0)
        img = etiquetar2(img, clOP, c, clOA, boxOA)

    return img


# ---------------------------
# 3) FASTAPI
# ---------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    result_img = CorrerModelo(img)

    _, buffer = cv2.imencode(".jpg", result_img)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return JSONResponse({
        "imagen_procesada": f"data:image/jpeg;base64,{img_base64}",
        "imagen_etiquetada": f"data:image/jpeg;base64,{img_base64}"
    })
