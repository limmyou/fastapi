from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import torch
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import io, time, traceback
from PIL import Image
# =========================================================
# FastAPI
# =========================================================
app = FastAPI(title="YOLO + DeepLabV3+ Unified API")

# =========================================================
# YOLO (단일 로딩, predict만 사용)
# =========================================================
YOLO_MODEL_PATH = "best2.pt"
print("🚀 Loading YOLO model...")
yolo_model = YOLO(YOLO_MODEL_PATH)
print("✅ YOLO model loaded")

# =========================================================
# DeepLabV3+
# =========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DLAB_MODEL_PATH = "best_dlab30.pth"

dlab_model = smp.DeepLabV3Plus(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
)
dlab_model.load_state_dict(torch.load(DLAB_MODEL_PATH, map_location=DEVICE))
dlab_model.to(DEVICE)
dlab_model.eval()
print("✅ DeepLabV3+ model loaded")

# =========================================================
# DeepLab 전처리
# =========================================================
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

def calculate_area(mask, shape):
    h, w = shape
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    area_pixels = int(mask.sum())
    area_ratio = area_pixels / (h * w) * 100
    area_cm2 = area_pixels * 0.0001
    return area_pixels, area_ratio, area_cm2

# =========================================================
# YOLO /detect
# =========================================================
@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        if not image_bytes:
            raise ValueError("빈 파일 업로드됨")

        # ✅ PIL로 안전하게 디코딩
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = np.array(image)

        start = time.time()

        results = yolo_model.predict(
            source=image,
            imgsz=640,
            conf=0.3,
            verbose=False
        )

        infer_ms = round((time.time() - start) * 1000, 2)

        preds = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            preds.append({
                "class_id": int(box.cls[0]),
                "confidence": round(float(box.conf[0]) * 100, 2),
                "box": [round(x1,2), round(y1,2), round(x2,2), round(y2,2)]
            })

        return {
            "object_count": len(preds),
            "inference_time_ms": infer_ms,
            "predictions": preds
        }

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

# =========================================================
# DeepLab /segment
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
