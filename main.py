from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
#from skimage import exposure,filters,feature
import cv2
import numpy as np
from ultralytics import YOLO
import io

from fastapi.middleware.cors import CORSMiddleware




# ---------------------------
# 1) CARGA DE MODELOS
# ---------------------------

modelrecorte = YOLO('recorte2.pt')
modeldetOP = YOLO('3clsOPfft.pt')
modeldetOA = YOLO('OAyoloR4cls5.pt')

# ---------------------------
# 2) FUNCIONES ORIGINALES
# ---------------------------

# Detectar pierna de una radiografía AP de rodilla
def yolorecorte(modelrecorte,img):
  results = modelrecorte(img)
  coor = []
  for result in results:
    for box in result.boxes:
      # Coordenadas
      x1, y1, x2, y2 = map(int, box.xyxy[0])
      coor.append([x1,y1,x2,y2])
  print(len(coor))
  print(coor)
  return coor

# Detectar osteoporosis
def yolodetOPCrop(modeldetOPfft,crop):
  
  if crop.ndim==3:
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
  #### fft
  f = np.fft.fft2(crop)
  fshift = np.fft.fftshift(f)
  # 3. Magnitud del espectro (log para mejor visualización)
  ms = 20 * np.log(np.abs(fshift) + 1)
  ms=ms.astype(np.uint8)
  #plt.imshow(ms)

  results = modeldetOPfft(ms)

  for result in results:
    cls = int(result.probs.top1)
    prob = float(result.probs.top1conf)

  return cls,prob

# detectar Osteoartritis
def yolodetOA(modeldet,impath,certeza):
  results = modeldet(impath)
  #results[0].show()
  cls = []
  prob = []
  for result in results:
    for box in result.boxes:
      # Coordenadas
      x1, y1, x2, y2 = map(int, box.xyxy[0])  # convertir a enteros
      #Conteo de detecciones de Op y normal
      conf = box.conf[0].item()
      if conf>certeza: #
        cls.append(int(box.cls))
        prob.append(conf)
  x=prob.index(np.max(prob))

  return cls[x],prob[x],x1, y1, x2, y2

def etiquetar2(imagen,clOP,xOP1,yOP1,xOP2,yOP2,clOA,xOA1, yOA1, xOA2, yOA2):
  color = (255, 0, 0) # Azul
  grosor = 2

  p1OP = (xOP1,yOP1)
  p2OP = (xOP2,yOP2)

  cv2.rectangle(imagen, p1OP, p2OP, color, grosor)
  if clOP==0:
    etiqueta='Sin osteoporosis'
  elif clOP==1:
    etiqueta='Osteopenia'  
  else:
    etiqueta='Osteoporosis'

  cv2.putText(imagen, etiqueta, (xOP1, yOP1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

  p1OA = (xOP1+xOA1,yOP1+yOA1)
  p2OA = (xOP1+xOA2,yOP1+yOA2)

  cv2.rectangle(imagen, p1OA, p2OA, color, grosor)
  if clOA==0:
    etiqueta='Sin Osteoartrosis'
  elif clOA==1:
    etiqueta='OA dudoso'
  elif clOA==2:
    etiqueta='OA leve'
  elif clOA==3:
    etiqueta='OA moderado'
  else:
    etiqueta='OA grave'

  cv2.putText(imagen, etiqueta, (xOP1+xOA1, yOP1+yOA1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
  return imagen


def CorrerModelo(path):
  certeza = 0
  img = cv2.imread(path)
  coor = yolorecorte(modelr1,img)

  for c in coor:
    crop = img[c[1]:c[3],c[0]:c[2],:]
    clOP,probOP=yolodetOPCrop(modeldetOPfft,crop)
    clOA,probOA,xOA1, yOA1, xOA2, yOA2 = yolodetOA(modeldetOA,crop,certeza)
    imagen = etiquetar2(img,clOP,c[0],c[1],c[2],c[3],clOA,xOA1, yOA1, xOA2, yOA2)
  return imagen


# ---------------------------
# 3) API FASTAPI
# ---------------------------

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Leer bytes
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Ejecutar modelo
    result_img = CorrerModelo(img)

    # Convertir a bytes JPG
    _, img_encoded = cv2.imencode(".jpg", result_img)
    return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/jpeg")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#uvicorn app:app --reload --port 8000  