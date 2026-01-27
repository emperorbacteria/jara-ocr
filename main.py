from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
from PIL import Image
import base64
import io
import re
import os
import threading

app = FastAPI(title="Jara OCR Service", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy initialization of PaddleOCR
ocr = None
ocr_lock = threading.Lock()
ocr_ready = False

def get_ocr():
    global ocr, ocr_ready
    if ocr is None:
        with ocr_lock:
            if ocr is None:
                print("Initializing PaddleOCR...")
                from paddleocr import PaddleOCR
                ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
                ocr_ready = True
                print("PaddleOCR ready!")
    return ocr

# Utility keywords for bill detection
UTILITY_KEYWORDS = {
    'airtime': 3, 'data': 3, 'electricity': 4, 'water': 3, 'gas': 3,
    'internet': 3, 'phone': 2, 'glo': 3, 'mtn': 3, 'airtel': 3, '9mobile': 3,
    'kwh': 4, 'meter': 4, 'recharge': 2, 'successful': 2, 'transaction': 2,
    'receipt': 2, 'paid': 2, 'credited': 2, 'debit': 2, 'transfer': 2,
    'units': 3, 'token': 3, 'vend': 3, 'prepaid': 3, 'postpaid': 3,
    'bank': 2, 'payment': 2, 'confirmed': 2
}


class OCRRequest(BaseModel):
    image_base64: str
    user_id: str = None


class OCRResponse(BaseModel):
    success: bool
    text: str
    amount: float
    bill_type: str
    transaction_id: str | None
    date: str | None
    time: str | None
    utility_score: int
    confidence: float


def preprocess_image(img_bytes: bytes) -> np.ndarray:
    """Preprocess image for better OCR"""
    # Load image
    img = Image.open(io.BytesIO(img_bytes))

    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Resize if too large
    max_dim = 1500
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    # Convert to numpy
    img_np = np.array(img)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Denoise
    img_cv = cv2.fastNlMeansDenoisingColored(img_cv, None, 10, 10, 7, 21)

    # Enhance contrast
    lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img_cv = cv2.merge([l, a, b])
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_LAB2BGR)

    return img_cv


def extract_text(img_cv: np.ndarray) -> tuple[str, float]:
    """Extract text using PaddleOCR"""
    ocr_instance = get_ocr()
    result = ocr_instance.ocr(img_cv, cls=True)

    if not result or not result[0]:
        return "", 0.0

    lines = []
    confidences = []

    for line in result[0]:
        if line and len(line) >= 2:
            text = line[1][0]
            conf = line[1][1]
            if conf > 0.5:
                lines.append(text)
                confidences.append(conf)

    full_text = '\n'.join(lines)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0

    return full_text, avg_conf


def extract_amount(text: str) -> float:
    """Extract monetary amount"""
    patterns = [
        r'(?:₦|NGN|N)\s*([\d,]+(?:\.\d{2})?)',
        r'(?:amount|total|paid|successful|credited)\s*[:\s]*(?:₦|NGN|N)?\s*([\d,]+(?:\.\d{2})?)',
        r'([\d,]+(?:\.\d{2})?)\s*(?:naira|NGN)',
    ]

    amounts = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                amount = float(match.replace(',', ''))
                if 10 <= amount <= 10000000:
                    amounts.append(amount)
            except ValueError:
                continue

    return max(amounts) if amounts else 0


def extract_transaction_id(text: str) -> str | None:
    """Extract transaction ID"""
    patterns = [
        r'(?:transaction\s*id|trans\s*id|ref|reference|txn)\s*[:\s]*([\w\d-]{6,25})',
        r'\b([A-Z0-9]{10,25})\b',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def extract_date_time(text: str) -> tuple[str | None, str | None]:
    """Extract date and time"""
    date_patterns = [
        r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})',
        r'(\d{1,2}\s+\w{3,9}\s+\d{2,4})',
    ]

    time_pattern = r'(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?)'

    date = None
    time = None

    for pattern in date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            date = match.group(1)
            break

    time_match = re.search(time_pattern, text, re.IGNORECASE)
    if time_match:
        time = time_match.group(1)

    return date, time


def detect_bill_type(text: str) -> str:
    """Detect type of bill"""
    text_lower = text.lower()

    if any(kw in text_lower for kw in ['airtime', 'recharge', 'top up', 'topup']):
        return 'airtime'
    elif any(kw in text_lower for kw in ['data', 'mb', 'gb', 'bundle']):
        return 'data'
    elif any(kw in text_lower for kw in ['electricity', 'kwh', 'meter', 'prepaid', 'phcn', 'eko', 'ikeja']):
        return 'electricity'
    elif any(kw in text_lower for kw in ['water', 'fctwb']):
        return 'water'
    elif any(kw in text_lower for kw in ['internet', 'wifi', 'broadband']):
        return 'internet'
    elif any(kw in text_lower for kw in ['gas', 'cooking']):
        return 'gas'
    else:
        return 'utility'


def calculate_utility_score(text: str) -> int:
    """Calculate utility bill confidence score"""
    text_lower = text.lower()
    return sum(
        weight for kw, weight in UTILITY_KEYWORDS.items()
        if kw in text_lower
    )


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "service": "Jara OCR Service",
        "version": "1.0.0",
        "ocr_engine": "PaddleOCR",
        "ocr_ready": ocr_ready
    }


@app.get("/warmup")
def warmup():
    """Trigger OCR initialization"""
    get_ocr()
    return {"status": "ok", "ocr_ready": True}


@app.post("/ocr", response_model=OCRResponse)
async def process_image(request: OCRRequest):
    """Process image and extract bill information"""
    try:
        # Decode base64 image
        try:
            # Handle data URL format
            if ',' in request.image_base64:
                image_data = request.image_base64.split(',')[1]
            else:
                image_data = request.image_base64

            img_bytes = base64.b64decode(image_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")

        # Preprocess
        img_cv = preprocess_image(img_bytes)

        # Extract text
        text, confidence = extract_text(img_cv)

        if not text:
            return OCRResponse(
                success=False,
                text="",
                amount=0,
                bill_type="unknown",
                transaction_id=None,
                date=None,
                time=None,
                utility_score=0,
                confidence=0
            )

        # Extract fields
        amount = extract_amount(text)
        transaction_id = extract_transaction_id(text)
        date, time = extract_date_time(text)
        bill_type = detect_bill_type(text)
        utility_score = calculate_utility_score(text)

        return OCRResponse(
            success=True,
            text=text,
            amount=amount,
            bill_type=bill_type,
            transaction_id=transaction_id,
            date=date,
            time=time,
            utility_score=utility_score,
            confidence=confidence
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
