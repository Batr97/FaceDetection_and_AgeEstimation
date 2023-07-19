from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import io
import imageio.v3 as iio
import cv2
from models.ViT_age_estimation import OpenEyesClassificator
from utils.image_utils import detect_img, detect_video


# Create the FastAPI app
app = FastAPI()

# Initialization of Vision Transformer model
ViT_model = OpenEyesClassificator('Batr97/ViT_ordinary')


@app.post("/predict/image/")
async def predict(file: UploadFile):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    image = np.array(image.convert('RGB'))
    image = detect_img(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    age_prediction = ViT_model.predict(inpIm=image)
    print('age_prediction', age_prediction)
    
    return JSONResponse(content={"age_prediction": age_prediction.item()})


@app.post("/predict/video/")
async def predict(file: UploadFile):
    contents = await file.read()
    video_stream = io.BytesIO(contents)
    video_cap = iio.imread(video_stream, index=None, format_hint=".mp4")
    image = detect_video(video_cap)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    age_prediction = ViT_model.predict(inpIm=image)
    print('age_prediction', age_prediction)

    return JSONResponse(content={"age_prediction": age_prediction.item()})
