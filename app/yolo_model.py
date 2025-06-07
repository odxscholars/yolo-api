import torch
from PIL import Image
from fastapi.responses import JSONResponse
import io
import ultralytics

# Load your YOLOv11 model
model = ultralytics.YOLO("yolov8s-seg.pt")  # Ensure you have the correct path to your model file
model.conf = 0.25  # Set confidence threshold
def predict(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    results = model(image)

    output = []
    for r in results:
        boxes = r.boxes
        masks = r.masks
        names = model.names

        preds = []
        for i, box in enumerate(boxes):
            pred = {
                "class": int(box.cls.cpu().numpy()[0]),
                "class_name": names[int(box.cls.cpu().numpy()[0])],
                "confidence": float(box.conf.cpu().numpy()[0]),
                "bbox": box.xyxy.cpu().numpy()[0].tolist()
            }
            if masks is not None:
                pred["mask"] = masks.data[i].cpu().numpy().tolist()
            preds.append(pred)
        output.extend(preds)

    return JSONResponse(content={"predictions": output})
