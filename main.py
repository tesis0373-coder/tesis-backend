from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO
import io

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
    for result in results:
        for box in result.boxes:
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
    for result in results:
        cls = int(result.probs.top1)
        prob = float(result.probs.top1conf)

    return cls, prob


def yolodetOA(model, img, certeza=0):
    results = model(img)
    cls, prob = [], []
    coords = []

    for result in results:
        for box in result.boxes:
            conf = box.conf[0].item()
            if conf > certeza:
                cls.append(int(box.cls))
                prob.append(conf)
                coords.append(tuple(map(int, box.xyxy[0])))

    if not prob:
        return None

    i = prob.index(max(prob))
    return cls[i], prob[i], *coords[i]


def etiquetar2(img, clOP, x1, y1, x2, y2, oa):
    color = (255, 0, 0)
    grosor = 2

    cv2.rectangle(img, (x1, y1), (x2, y2), color, grosor)

    etiquetas_op = ["Sin osteoporosis", "Osteopenia", "Osteoporosis"]
    cv2.putText(
        img,
        etiquetas_op[clOP],
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    if oa:
        clOA, probOA, ox1, oy1, ox2, oy2 = oa
        cv2.rectangle(
            img,
            (x1 + ox1, y1 + oy1),
            (x1 + ox2, y1 + oy2),
            color,
            grosor,
        )

    return img


def CorrerModelo(img):
    coor = yolorecorte(modelrecorte, img)

    for c in coor:
        crop = img[c[1]:c[3], c[0]:c[2]]
        clOP, _ = yolodetOPCrop(modeldetOP, crop)
        oa = yolodetOA(modeldetOA, crop)
        img = etiquetar2(img, clOP, c[0], c[1], c[2], c[3], oa)

    return img

# ---------------------------
# 3) API FASTAPI
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
async def predict(request: Request):
    form = await request.form()

    file = None
    for value in form.values():
        if hasattr(value, "filename"):
            file = value
            break

    if file is None:
        return JSONResponse(status_code=400, content={"error": "No file received"})

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    result = CorrerModelo(img)

    _, img_encoded = cv2.imencode(".jpg", result)
    return StreamingResponse(
        io.BytesIO(img_encoded.tobytes()),
        media_type="image/jpeg"
    )
