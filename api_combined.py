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

# =========================================================
# FastAPI
# =========================================================
app = FastAPI(title="YOLO + DeepLabV3+ Unified API")

# =========================================================
# YOLO (lazy load, 단일 방식)
# =========================================================
YOLO_MODEL_PATH = "best2.pt"
_yolo = None

def get_yolo():
    global _yolo
    if _yolo is None:
        print("🚀 YOLO model loading...")
        _yolo = YOLO(YOLO_MODEL_PATH)
        _yolo.fuse()
    return _yolo

is_busy = False

@app.get("/status")
def status():
    return {"status": "busy" if is_busy else "idle"}

# =========================================================
# DeepLabV3+
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
        mask = torch.sigmoid(pred).squeeze().cpu().numpy()
        return (mask > 0.38).astype(np.uint8)

def calculate_area(mask, orig_shape):
    h, w = orig_shape
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    pixels = int(mask.sum())
    ratio = pixels / (h * w) * 100
    area_cm2 = pixels * 0.0001
    return pixels, ratio, area_cm2

# =========================================================
# YOLO /detect  ✅ (최종 안정)
# =========================================================
@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    global is_busy
    try:
        is_busy = True

        # 1️⃣ 파일 1회 읽기
        image_bytes = await file.read()

        # 2️⃣ PIL Image ONLY
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        yolo = get_yolo()

        start = time.time()
        results = yolo(image, conf=0.3, imgsz=640, verbose=False)
        infer_ms = round((time.time() - start) * 1000, 2)

        preds = []
        for box in results[0].boxes:
            preds.append({
                "class_id": int(box.cls[0]),
                "confidence": round(float(box.conf[0]) * 100, 2),
                "box": [round(x, 2) for x in box.xyxy[0].tolist()]
            })

        return {
            "object_count": len(preds),
            "inference_time_ms": infer_ms,
            "predictions": preds
        }

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        is_busy = False

# =========================================================
# DeepLab /segment (이건 그대로 OK)
# =========================================================
@app.post("/segment")
async def segment_area(file: UploadFile = File(...)):
    try:
        start = time.time()

        contents = await file.read()
        npimg = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        orig_shape = image.shape[:2]
        image_t = preprocess(image)
        mask = predict_mask(image_t)

        pixels, ratio, area_cm2 = calculate_area(mask, orig_shape)
        infer_ms = round((time.time() - start) * 1000, 2)

        return {
            "area_count": pixels,
            "area_ratio_percent": round(ratio, 2),
            "area_cm2_assumed": round(area_cm2, 2),
            "inference_time_ms": infer_ms
        }

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
