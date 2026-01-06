from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import torch
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import io, time, traceback

app = FastAPI(title="YOLO + DeepLabV3+ Unified API")

# =========================================================
# YOLO (lazy load, 단일 방식)
# =========================================================
YOLO_MODEL_PATH = "best2.pt"
yolo_model = None

def get_yolo():
    global yolo_model
    if yolo_model is None:
        print("🚀 YOLO model loading...")
        yolo_model = YOLO(YOLO_MODEL_PATH)
    return yolo_model

# =========================================================
# DeepLabV3+ (기존 그대로 OK)
# =========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DLAB_MODEL_PATH = "best_dlab30.pth"

def load_dlab_model():
    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    )
    model.load_state_dict(torch.load(DLAB_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

dlab_model = load_dlab_model()
print("✅ DeepLabV3+ model loaded")

val_tf = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

def preprocess(image):
    aug = val_tf(image=image)
    return aug["image"].unsqueeze(0).to(DEVICE)

def predict_mask(image_tensor):
    with torch.no_grad():
        pred = dlab_model(image_tensor)
        return (torch.sigmoid(pred).squeeze().cpu().numpy() > 0.38).astype(np.uint8)

def calculate_area(pred_mask, orig_shape):
    h, w = orig_shape
    mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    area_pixels = int(mask.sum())
    area_ratio = area_pixels / (h * w) * 100
    area_cm2 = area_pixels * 0.0001
    return area_pixels, area_ratio, area_cm2

# =========================================================
# YOLO /detect (🔥 여기만 보면 됨)
# =========================================================
@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    try:
        yolo = get_yolo()

        # 1️⃣ 파일 읽기
        image_bytes = await file.read()

        # 2️⃣ PIL → numpy (RGB, uint8)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image, dtype=np.uint8)

        # 3️⃣ YOLO predict (이 방식만!)
        start = time.time()
        results = yolo.predict(
            source=image_np,
            imgsz=640,
            conf=0.3,
            verbose=False
        )
        infer_ms = round((time.time() - start) * 1000, 2)

        predictions = []
        for box in results[0].boxes:
            predictions.append({
                "class_id": int(box.cls[0]),
                "confidence": round(float(box.conf[0]) * 100, 2),
                "box": [round(v, 2) for v in box.xyxy[0].tolist()]
            })

        return {
            "object_count": len(predictions),
            "inference_time_ms": infer_ms,
            "predictions": predictions
        }

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

# =========================================================
# DeepLab /segment (기존 유지)
# =========================================================
@app.post("/segment")
async def segment_area(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        npimg = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_t = preprocess(image)
        mask = predict_mask(image_t)

        area_pixels, area_ratio, area_cm2 = calculate_area(mask, image.shape[:2])

        return {
            "area_count": area_pixels,
            "area_ratio_percent": round(area_ratio, 2),
            "area_cm2_assumed": round(area_cm2, 2)
        }

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
