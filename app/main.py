from fastapi import FastAPI, File, UploadFile
from app.yolo_model import predict

app = FastAPI()

@app.post("/predict")
async def detect_objects(file: UploadFile = File(...)):
    image_bytes = await file.read()
    predictions = predict(image_bytes)
    return {"predictions": predictions}

@app.get("/")
async def root():
    return {"message": "Welcome to the YOLOv11 Object Detection API. Use /predict to upload an image."}
