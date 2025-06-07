from fastapi import FastAPI, File, UploadFile
from app.yolo_model import predict

app = FastAPI()

@app.post("/predict")
async def detect_objects(file: UploadFile = File(...)):
    image_bytes = await file.read()
    predictions = predict(image_bytes)
    return {"predictions": predictions}
