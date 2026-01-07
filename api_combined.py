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
# Busy/Lock (동시 요청 방지)
# =========================================================
busy_lock = asyncio.Lock()
is_busy = False

@app.get("/status")
def status():
    return {"status": "busy" if is_busy else "idle"}

# =========================================================
# 공통: 업로드 파일 -> 이미지 디코딩 (항상 np.ndarray 보장)
#   - 반환: rgb(np.uint8 HWC), bgr(np.uint8 HWC)
# =========================================================
def decode_upload_image(image_bytes: bytes):
    if not image_bytes or len(image_bytes) < 10:
        raise ValueError("업로드된 파일이 비어있거나 너무 작습니다.")

    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"PIL 이미지 오픈 실패: {e}")

    rgb = np.array(pil_img, dtype=np.uint8)

    if not isinstance(rgb, np.ndarray):
        raise ValueError("RGB 변환 실패: numpy array 아님")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"RGB shape 이상: {rgb.shape}")

    # contiguous 보장
    rgb = np.ascontiguousarray(rgb)

    # ✅ OpenCV 사용 금지 → numpy slicing
    bgr = rgb[:, :, ::-1].copy()

    return rgb, bgr

# =========================================================
# YOLO (lazy load + fuse 1회)
# =========================================================
YOLO_MODEL_PATH = "best2.pt"
_yolo_model = None

def get_yolo():
    global _yolo_model
    if _yolo_model is None:
        print("🚀 Loading YOLO model...")
        m = YOLO(YOLO_MODEL_PATH)
        # fuse는 되면 하고, 안 되면 패스(환경에 따라 예외 가능)
        try:
            m.fuse()
        except Exception as e:
            print(f"⚠️ YOLO fuse skipped: {e}")
        _yolo_model = m
        print("✅ YOLO model loaded")
    return _yolo_model

# =========================================================
# DeepLabV3+ 모델 로드
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

# DeepLab 전처리
val_tf = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

def preprocess(image_rgb_uint8: np.ndarray):
    aug = val_tf(image=image_rgb_uint8)
    return aug["image"].unsqueeze(0).to(DEVICE)

def predict_mask(image_tensor):
    with torch.no_grad():
        pred = dlab_model(image_tensor)
        pred_mask = torch.sigmoid(pred).squeeze().cpu().numpy()
        return (pred_mask > 0.38).astype(np.uint8)

def calculate_area(pred_mask, orig_shape_hw):
    orig_h, orig_w = orig_shape_hw
    pred_mask_resized = cv2.resize(pred_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    area_pixels = int(np.sum(pred_mask_resized))
    total_pixels = int(orig_h * orig_w)
    area_ratio = (area_pixels / total_pixels) * 100 if total_pixels > 0 else 0.0
    area_cm2 = area_pixels * 0.0001  # 가정
    return area_pixels, area_ratio, area_cm2

# =========================================================
# YOLO /detect endpoint
# =========================================================
@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    global is_busy
    async with busy_lock:
        try:
            is_busy = True

            image_bytes = await file.read()
            # 디코딩: rgb/bgr 둘 다 확보 (YOLO는 bgr로 넣는게 가장 안전)
            rgb, bgr = decode_upload_image(image_bytes)

            # 디버그 (로그에 찍혀서 타입 확인 가능)
            print(f"DEBUG upload={file.filename} bytes={len(image_bytes)} "
                  f"bgr.shape={bgr.shape} dtype={bgr.dtype} contiguous={bgr.flags['C_CONTIGUOUS']}")

            yolo = get_yolo()

            start_time = time.time()
            # ✅ Ultralytics는 predict로 고정
            results = yolo.predict(
                source=bgr,     # numpy.ndarray (H,W,3) uint8
                imgsz=640,
                conf=0.3,
                verbose=False
            )
            inference_time = round((time.time() - start_time) * 1000, 2)

            predictions = []
            object_count = 0

            if results and len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    object_count += 1
                    predictions.append({
                        "class_id": cls_id,
                        "confidence": round(conf * 100, 2),
                        "box": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
                    })

            return {
                "filename": file.filename,
                "object_count": object_count,
                "inference_time_ms": inference_time,
                "predictions": predictions
            }

        except Exception as e:
            traceback.print_exc()
            return JSONResponse(status_code=500, content={"error": str(e)})

        finally:
            is_busy = False

# =========================================================
# DeepLab /segment endpoint
# =========================================================
@app.post("/segment")
async def segment_area(file: UploadFile = File(...)):
    try:
        start_time = time.time()

        image_bytes = await file.read()
        rgb, _ = decode_upload_image(image_bytes)
        orig_shape = rgb.shape[:2]  # (h, w)

        image_t = preprocess(rgb)
        pred_mask = predict_mask(image_t)

        area_pixels, area_ratio, area_cm2 = calculate_area(pred_mask, orig_shape)
        inference_time_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "model": "DeepLabV3+",
            "filename": file.filename,
            "area_count": int(area_pixels),
            "area_ratio_percent": round(area_ratio, 2),
            "area_cm2_assumed": round(area_cm2, 2),
            "inference_time_ms": inference_time_ms
        }

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
