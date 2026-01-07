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
import io, time, traceback, asyncio

# =========================================================
# FastAPI
# =========================================================
app = FastAPI(title="YOLO + DeepLabV3+ Unified API")

# =========================================================
# Busy lock (동시 요청 방지)
# =========================================================
busy_lock = asyncio.Lock()
is_busy = False

@app.get("/status")
def status():
    return {"status": "busy" if is_busy else "idle"}

# =========================================================
# DeepLab 전용: 업로드 → RGB numpy
# =========================================================
def decode_upload_image(image_bytes: bytes) -> np.ndarray:
    if not image_bytes or len(image_bytes) < 10:
        raise ValueError("업로드된 파일이 비어있거나 너무 작습니다.")

    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"PIL 이미지 오픈 실패: {e}")

    rgb = np.array(pil_img, dtype=np.uint8, copy=True)

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"RGB shape 오류: {rgb.shape}")

    return np.ascontiguousarray(rgb)

# =========================================================
# YOLO (PIL 전용, 절대 numpy/cv2 금지)
# =========================================================
YOLO_MODEL_PATH = "best2.pt"
_yolo = None

def get_yolo():
    global _yolo
    if _yolo is None:
        print("🚀 Loading YOLO model...")
        model = YOLO(YOLO_MODEL_PATH)
        try:
            model.fuse()
        except:
            pass
        _yolo = model
        print("✅ YOLO model loaded")
    return _yolo

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

val_tf = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

def preprocess(rgb: np.ndarray):
    aug = val_tf(image=rgb)
    return aug["image"].unsqueeze(0).to(DEVICE)

def predict_mask(tensor):
    with torch.no_grad():
        p = torch.sigmoid(dlab_model(tensor)).squeeze().cpu().numpy()
        return (p > 0.38).astype(np.uint8)

# =========================================================
# YOLO /detect (🔥 PIL만 사용)
# =========================================================
@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    global is_busy
    async with busy_lock:
        try:
            is_busy = True

            image_bytes = await file.read()

            # ✅ YOLO는 PIL Image 그대로
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            yolo = get_yolo()

            start = time.time()
            results = yolo.predict(
                source=pil_img,
                imgsz=640,
                conf=0.3,
                verbose=False
            )
            infer_ms = round((time.time() - start) * 1000, 2)

            predictions = []
            if results and results[0].boxes is not None:
                for b in results[0].boxes:
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    predictions.append({
                        "class_id": int(b.cls[0]),
                        "confidence": round(float(b.conf[0]) * 100, 2),
                        "box": [round(x1,2), round(y1,2), round(x2,2), round(y2,2)]
                    })

            return {
                "filename": file.filename,
                "object_count": len(predictions),
                "inference_time_ms": infer_ms,
                "predictions": predictions
            }

        except Exception as e:
            traceback.print_exc()
            return JSONResponse(status_code=500, content={"error": str(e)})

        finally:
            is_busy = False

# =========================================================
# DeepLab /segment (numpy + cv2)
# =========================================================
@app.post("/segment")
async def segment_area(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        rgb = decode_upload_image(image_bytes)

        tensor = preprocess(rgb)
        mask = predict_mask(tensor)

        h, w = rgb.shape[:2]
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        area_px = int(mask.sum())
        area_ratio = area_px / (h * w) * 100

        return {
            "filename": file.filename,
            "area_pixels": area_px,
            "area_ratio_percent": round(area_ratio, 2)
        }

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
