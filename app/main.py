import cv2 as cv
from fastapi import FastAPI, UploadFile, File
import numpy as np
from utils import *
from model import *
from preprocessing import *

app = FastAPI()

model = setup_model()

@app.post("/register")
async def register_embed(file: UploadFile = File(...), labels: str = ''):
    image = await file.read()
    nparr = np.fromstring(image, np.uint8)
    f = cv.imdecode(nparr, cv.IMREAD_COLOR)
    transform_norm = transforms.Compose([transforms.ToTensor()])
    img_normalized = transform_norm(f).float()
    img_normalized = img_normalized.unsqueeze_(0)
    embedding = Feature_Extractor(img_normalized, model)
    register_embedding(embedding.detach().squeeze().numpy(), labels)
    return {
        "message": "success",
    }

@app.post("/recognize")
async def recognize_embed(file: UploadFile = File(...), threshold: float = 0.5):
    image = await file.read()
    nparr = np.fromstring(image, np.uint8)
    f = cv.imdecode(nparr, cv.IMREAD_COLOR)
    transform_norm = transforms.Compose([transforms.ToTensor()])
    img_normalized = transform_norm(f).float()
    img_normalized = img_normalized.unsqueeze_(0)
    embedding = Feature_Extractor(img_normalized, model)
    labels = recognize_embedding(embedding.detach().squeeze().numpy(), threshold=threshold)
    return {
        "label": labels,
    }