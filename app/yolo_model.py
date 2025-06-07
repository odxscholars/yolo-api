import torch
from PIL import Image
import io

# Load your YOLOv11 model
model = torch.hub.load('WongKinYiu/yolov11', 'custom', path='yolov11.pt')  # change path

def predict(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    results = model(image, size=640)
    return results.pandas().xyxy[0].to_dict(orient="records")  # bbox, conf, class, etc.
