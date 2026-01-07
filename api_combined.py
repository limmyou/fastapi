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

app = FastAPI(title="YOLO + DeepLabV3+ Unified API")

# =========================================================
# Busy lock
# =========================================================
busy_lock = asyncio.Lock()
is_busy = False

@app.get("/status")
def status():
    return {"status": "busy" if is_busy else "idle"}

# =========================================================
# ✅ 업로드 이미지 디코더 (이 이름으로 고정)
# =========================================================
def decode_upload_image(image_bytes: bytes):
    if not image_bytes or len(image_bytes) < 10:
        raise ValueError("빈 파일")

    # PIL 이미지 읽기
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # numpy 배열로 변환
    rgb = np.array(pil_img, dtype=np.uint8)

    # 디버그: RGB 배열 정보 확인
    print(f"DEBUG: RGB shape={rgb.shape}, dtype={rgb.dtype}")

    # RGB 배열이 3D 배열이고 채널 수가 3인지 확인
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"RGB shape 오류: {rgb.shape}")

    # OpenCV에서 처리 가능한 연속된 메모리 배열로 변환
    rgb = np.ascontiguousarray(rgb)

    # BGR로 변환
    try:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise ValueError(f"cv2.cvtColor 오류: {e}")
    
    bgr = np.ascontiguousarray(bgr)  # OpenCV의 추가적인 요구 사항

    return rgb, bgr

# =========================================================
# YOLO
# =========================================================
YOLO_MODEL_PATH = "best2.pt"
_yolo = None

def get_yolo():
    global _yolo
    if _yolo is None:
        print("🚀 Loading YOLO model...")
        _yolo = YOLO(YOLO_MODEL_PATH)
        print("✅ YOLO model loaded")
    return _yolo

# =========================================================
# DeepLabV3+
# =========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

dlab_model = smp.DeepLabV3Plus(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
)
dlab_model.load_state_dict(torch.load("best_dlab30.pth", map_location=DEVICE))
dlab_model.to(DEVICE)
dlab_model.eval()
print("✅ DeepLabV3+ model loaded")

val_tf = A.Compose([
    A.Resize(512, 512),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    ToTensorV2(),
])

def preprocess(rgb):
    aug = val_tf(image=rgb)
    return aug["image"].unsqueeze(0).to(DEVICE)

def predict_mask(t):
    with torch.no_grad():
        p = torch.sigmoid(dlab_model(t)).squeeze().cpu().numpy()
        return (p > 0.38).astype(np.uint8)

# =========================================================
# ✅ YOLO /detect (중요)
# =========================================================
@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    global is_busy
    async with busy_lock:
        try:
            is_busy = True

            image_bytes = await file.read()
            rgb, bgr = decode_upload_image(image_bytes)

            print(f"DEBUG bgr.shape={bgr.shape}, dtype={bgr.dtype}")

            yolo = get_yolo()

            start = time.time()
            # 🔥 반드시 리스트로 전달
            results = yolo([bgr], imgsz=640, conf=0.3, verbose=False)
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
                "filename": file.filename,
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
# DeepLab /segment
# =========================================================
@app.post("/segment")
async def segment_area(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        rgb, _ = decode_upload_image(image_bytes)

        t = preprocess(rgb)
        mask = predict_mask(t)

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
