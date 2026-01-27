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

# PaddleOCR initialization (models pre-downloaded in Docker build)
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
                # Use optimized settings for lower memory usage
                ocr = PaddleOCR(
                    use_angle_cls=False,  # Skip angle classification to save memory
                    lang='en'
                )
                ocr_ready = True
                print("PaddleOCR ready!")
    return ocr

def init_ocr_on_startup():
    """Initialize OCR on app startup in a background thread"""
    import time
    time.sleep(2)  # Wait for app to fully start
    try:
        get_ocr()
        print("OCR pre-initialized successfully")
    except Exception as e:
        print(f"OCR initialization error: {e}")

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
    currency: str  # ISO currency code (USD, EUR, GBP, NGN, etc.) or UNKNOWN
    amount_usd: float  # Converted to USD for JRA calculation
    bill_type: str
    transaction_id: str | None
    date: str | None
    time: str | None
    utility_score: int
    confidence: float
    supported_currencies: list[str] = []  # List of detected currencies


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
    result = ocr_instance.ocr(img_cv)

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


# Comprehensive global currency support
# Symbol to currency code mapping
CURRENCY_SYMBOLS = {
    # Major currencies
    '$': 'USD', '€': 'EUR', '£': 'GBP', '₦': 'NGN', '¥': 'JPY', '₹': 'INR',
    '₽': 'RUB', '₩': 'KRW', '฿': 'THB', '₱': 'PHP', '₫': 'VND', '₴': 'UAH',
    '₺': 'TRY', '₼': 'AZN', '₾': 'GEL', '₸': 'KZT', '₿': 'BTC',
    # Multi-char symbols
    'R$': 'BRL', 'RM': 'MYR', 'Rp': 'IDR', 'Rs': 'INR', 'Rs.': 'INR',
    'kr': 'SEK', 'Kr': 'SEK', 'Kč': 'CZK', 'zł': 'PLN', 'Ft': 'HUF',
    'lei': 'RON', 'лв': 'BGN', 'din': 'RSD', 'kn': 'HRK',
    # Asian
    '元': 'CNY', '円': 'JPY', '원': 'KRW', '₮': 'MNT',
    # African
    'R': 'ZAR', 'KSh': 'KES', 'GH₵': 'GHS', '₵': 'GHS', 'TSh': 'TZS',
    'USh': 'UGX', 'CFA': 'XOF', 'FCFA': 'XAF', 'DH': 'MAD', 'DA': 'DZD',
    'E£': 'EGP', 'LE': 'EGP',
    # Middle East
    'ر.س': 'SAR', 'د.إ': 'AED', 'ر.ق': 'QAR', 'د.ك': 'KWD', 'د.ب': 'BHD',
    'ر.ع': 'OMR', 'د.أ': 'JOD', '₪': 'ILS',
    # Americas
    'C$': 'CAD', 'A$': 'AUD', 'NZ$': 'NZD', 'MX$': 'MXN', 'AR$': 'ARS',
    'CL$': 'CLP', 'CO$': 'COP', 'S/.': 'PEN',
}

# All supported currency codes
CURRENCY_CODES = [
    # Major
    'USD', 'EUR', 'GBP', 'NGN', 'JPY', 'CNY', 'INR', 'CAD', 'AUD', 'CHF',
    # Asia Pacific
    'KRW', 'SGD', 'HKD', 'TWD', 'THB', 'MYR', 'IDR', 'PHP', 'VND', 'PKR',
    'BDT', 'LKR', 'NPR', 'MMK', 'KHR', 'LAK', 'BND', 'MNT',
    # Europe
    'SEK', 'NOK', 'DKK', 'PLN', 'CZK', 'HUF', 'RON', 'BGN', 'HRK', 'RSD',
    'UAH', 'RUB', 'BYN', 'MDL', 'ALL', 'MKD', 'BAM', 'ISK', 'TRY', 'GEL',
    'AZN', 'AMD', 'KZT', 'UZS', 'KGS', 'TJS', 'TMT',
    # Americas
    'MXN', 'BRL', 'ARS', 'CLP', 'COP', 'PEN', 'VES', 'UYU', 'PYG', 'BOB',
    'GTQ', 'HNL', 'NIO', 'CRC', 'PAB', 'DOP', 'CUP', 'JMD', 'TTD', 'BBD',
    'BSD', 'BZD', 'GYD', 'SRD', 'HTG', 'AWG', 'ANG', 'XCD',
    # Middle East
    'SAR', 'AED', 'QAR', 'KWD', 'BHD', 'OMR', 'JOD', 'LBP', 'SYP', 'IQD',
    'IRR', 'YER', 'ILS', 'EGP', 'AFN',
    # Africa
    'ZAR', 'KES', 'GHS', 'TZS', 'UGX', 'RWF', 'BIF', 'ETB', 'SOS', 'DJF',
    'ERN', 'SDG', 'SSP', 'XOF', 'XAF', 'MAD', 'DZD', 'TND', 'LYD', 'MUR',
    'SCR', 'MGA', 'MWK', 'ZMW', 'BWP', 'NAD', 'SZL', 'LSL', 'AOA', 'CDF',
    'GMD', 'GNF', 'LRD', 'SLL', 'CVE', 'STN', 'MZN', 'ZWL',
    # Oceania
    'NZD', 'FJD', 'PGK', 'SBD', 'VUV', 'WST', 'TOP', 'XPF',
    # Crypto
    'BTC', 'ETH', 'USDT', 'USDC',
]

# Exchange rates to USD (approximate, for offline calculation)
# Values: 1 unit of currency = X USD
EXCHANGE_RATES_TO_USD = {
    # Major currencies
    'USD': 1.0, 'EUR': 1.08, 'GBP': 1.27, 'CHF': 1.12, 'JPY': 0.0067,
    'CNY': 0.14, 'INR': 0.012, 'CAD': 0.74, 'AUD': 0.65, 'NZD': 0.61,
    # Nigerian Naira
    'NGN': 0.000625,  # 1 NGN = 0.000625 USD (1600 NGN per USD)
    # Asian
    'KRW': 0.00075, 'SGD': 0.74, 'HKD': 0.13, 'TWD': 0.031, 'THB': 0.028,
    'MYR': 0.22, 'IDR': 0.000063, 'PHP': 0.018, 'VND': 0.00004, 'PKR': 0.0036,
    'BDT': 0.0091, 'LKR': 0.003, 'NPR': 0.0075,
    # European
    'SEK': 0.095, 'NOK': 0.092, 'DKK': 0.14, 'PLN': 0.25, 'CZK': 0.043,
    'HUF': 0.0027, 'RON': 0.22, 'BGN': 0.55, 'HRK': 0.14, 'UAH': 0.027,
    'RUB': 0.011, 'TRY': 0.031, 'ISK': 0.0072,
    # Middle East
    'SAR': 0.27, 'AED': 0.27, 'QAR': 0.27, 'KWD': 3.25, 'BHD': 2.65,
    'OMR': 2.60, 'JOD': 1.41, 'ILS': 0.27, 'EGP': 0.032, 'LBP': 0.000011,
    # African
    'ZAR': 0.055, 'KES': 0.0078, 'GHS': 0.083, 'TZS': 0.00039, 'UGX': 0.00027,
    'XOF': 0.0016, 'XAF': 0.0016, 'MAD': 0.10, 'DZD': 0.0074, 'TND': 0.32,
    # Americas
    'MXN': 0.058, 'BRL': 0.20, 'ARS': 0.0012, 'CLP': 0.0011, 'COP': 0.00025,
    'PEN': 0.27,
    # Default for unknown
    'DEFAULT': 1.0,
}

# Build regex patterns for all currencies
def build_currency_patterns():
    """Build comprehensive currency detection patterns"""
    patterns = {}

    # Symbol-based patterns (high priority)
    for symbol, code in CURRENCY_SYMBOLS.items():
        if code not in patterns:
            patterns[code] = []
        # Escape special regex chars in symbol
        escaped = re.escape(symbol)
        patterns[code].append(rf'{escaped}\s*([\d,]+(?:\.\d{{1,2}})?)')
        patterns[code].append(rf'([\d,]+(?:\.\d{{1,2}})?)\s*{escaped}')

    # Code-based patterns
    for code in CURRENCY_CODES:
        if code not in patterns:
            patterns[code] = []
        patterns[code].append(rf'(?:{code})\s*([\d,]+(?:\.\d{{1,2}})?)')
        patterns[code].append(rf'([\d,]+(?:\.\d{{1,2}})?)\s*(?:{code})')

    # Add generic amount patterns for NGN (common Nigerian format)
    if 'NGN' in patterns:
        patterns['NGN'].extend([
            r'(?:amount|total|paid|successful|credited)\s*[:\s]*(?:₦|NGN|N)?\s*([\d,]+(?:\.\d{1,2})?)',
            r'(?:naira)\s*([\d,]+(?:\.\d{1,2})?)',
            r'([\d,]+(?:\.\d{1,2})?)\s*(?:naira)',
        ])

    return patterns

CURRENCY_PATTERNS = build_currency_patterns()


def get_amount_range(currency: str) -> tuple[float, float]:
    """Get valid amount range for a currency based on its typical values"""
    # High-value currencies (1 unit > $1 USD)
    high_value = ['KWD', 'BHD', 'OMR', 'JOD', 'GBP', 'EUR', 'CHF', 'USD', 'CAD', 'AUD', 'SGD', 'BTC', 'ETH']
    # Medium-value currencies
    medium_value = ['SAR', 'AED', 'QAR', 'MYR', 'BRL', 'PLN', 'ILS', 'NZD', 'HKD', 'CNY']
    # Low-value currencies (need larger amounts)
    low_value = ['NGN', 'KRW', 'JPY', 'IDR', 'VND', 'IRR', 'LBP', 'VES', 'ZWL']

    if currency in high_value:
        return (0.01, 1000000)  # $0.01 to $1M
    elif currency in medium_value:
        return (0.1, 10000000)  # Slightly higher minimum
    elif currency in low_value:
        return (1, 100000000000)  # Large amounts common
    else:
        return (0.01, 100000000)  # Default range


def extract_amount_with_currency(text: str) -> tuple[float, str]:
    """Extract monetary amount and detect currency from any global currency"""
    results = {}  # currency -> list of (amount, confidence_score)

    for currency, patterns in CURRENCY_PATTERNS.items():
        min_amt, max_amt = get_amount_range(currency)
        amounts = []

        for i, pattern in enumerate(patterns):
            try:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    try:
                        # Handle both comma and period as thousand separators
                        cleaned = match.replace(',', '').replace(' ', '')
                        amount = float(cleaned)
                        if min_amt <= amount <= max_amt:
                            # Higher score for symbol-based matches (first patterns)
                            confidence = 10 - min(i, 5)
                            amounts.append((amount, confidence))
                    except ValueError:
                        continue
            except re.error:
                continue

        if amounts:
            # Sort by confidence and take highest amount among top confidence
            amounts.sort(key=lambda x: (-x[1], -x[0]))
            results[currency] = amounts

    if not results:
        # Try to find any number that looks like a bill amount
        generic_pattern = r'(?:total|amount|paid|sum|balance)[:\s]*([\\d,]+(?:\\.\\d{1,2})?)'
        matches = re.findall(generic_pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                amount = float(match.replace(',', ''))
                if 1 <= amount <= 100000000:
                    return amount, 'UNKNOWN'
            except ValueError:
                continue
        return 0, 'UNKNOWN'

    # Priority order for currency detection
    priority_order = [
        # Specific symbols first (high confidence)
        'USD', 'EUR', 'GBP', 'JPY', 'CNY', 'INR', 'CHF', 'CAD', 'AUD',
        # Regional currencies
        'NGN', 'KES', 'GHS', 'ZAR', 'EGP',  # Africa
        'SAR', 'AED', 'QAR', 'KWD', 'ILS',  # Middle East
        'BRL', 'MXN', 'ARS', 'COP', 'CLP',  # Americas
        'KRW', 'SGD', 'HKD', 'THB', 'MYR', 'IDR', 'PHP', 'VND',  # Asia
        'SEK', 'NOK', 'DKK', 'PLN', 'CZK', 'TRY', 'RUB',  # Europe
    ]

    # Find the best match
    best_currency = None
    best_amount = 0
    best_confidence = 0

    for currency in priority_order:
        if currency in results:
            top_match = results[currency][0]
            if top_match[1] > best_confidence or (top_match[1] == best_confidence and top_match[0] > best_amount):
                best_currency = currency
                best_amount = top_match[0]
                best_confidence = top_match[1]

    # If none from priority, use first found
    if not best_currency and results:
        best_currency = list(results.keys())[0]
        best_amount = results[best_currency][0][0]

    return best_amount, best_currency or 'UNKNOWN'


def convert_to_usd(amount: float, currency: str) -> float:
    """Convert amount to USD using exchange rates"""
    if amount <= 0:
        return 0

    if currency == 'USD':
        return amount

    rate = EXCHANGE_RATES_TO_USD.get(currency, EXCHANGE_RATES_TO_USD.get('DEFAULT', 1.0))
    return amount * rate


def extract_amount(text: str) -> float:
    """Extract monetary amount (legacy function for compatibility)"""
    amount, _ = extract_amount_with_currency(text)
    return amount


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
                currency="UNKNOWN",
                amount_usd=0,
                bill_type="unknown",
                transaction_id=None,
                date=None,
                time=None,
                utility_score=0,
                confidence=0,
                supported_currencies=CURRENCY_CODES[:20]  # Return top 20 for reference
            )

        # Extract fields
        amount, currency = extract_amount_with_currency(text)
        amount_usd = convert_to_usd(amount, currency)
        transaction_id = extract_transaction_id(text)
        date, time = extract_date_time(text)
        bill_type = detect_bill_type(text)
        utility_score = calculate_utility_score(text)

        return OCRResponse(
            success=True,
            text=text,
            amount=amount,
            currency=currency,
            amount_usd=round(amount_usd, 4),
            bill_type=bill_type,
            transaction_id=transaction_id,
            date=date,
            time=time,
            utility_score=utility_score,
            confidence=confidence,
            supported_currencies=CURRENCY_CODES[:20]
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def startup_event():
    """Initialize OCR in background on startup"""
    thread = threading.Thread(target=init_ocr_on_startup, daemon=True)
    thread.start()


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
