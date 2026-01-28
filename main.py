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
import traceback

app = FastAPI(title="Jara OCR Service", version="1.1.0")

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
                # Use optimized settings for better accuracy (multilingual)
                ocr = PaddleOCR(
                    use_angle_cls=True,  # Enable angle classification for rotated text
                    lang='en',  # English base (handles most receipts globally)
                    det_db_thresh=0.15,  # Very low threshold for max text detection
                    det_db_box_thresh=0.3,  # Lower threshold for box detection
                    det_db_unclip_ratio=2.0,  # Expand text boxes more
                    rec_batch_num=10,  # Process more text boxes at once
                    use_space_char=True,  # Better handling of spaces
                    drop_score=0.2,  # Very low drop threshold to keep more text
                )
                ocr_ready = True
                print("PaddleOCR ready with enhanced detection!")
    return ocr

def init_ocr_on_startup():
    """Initialize OCR on app startup in a background thread"""
    import time
    time.sleep(10)  # Wait longer for app to fully start and pass healthcheck
    try:
        print("Starting OCR pre-initialization...")
        get_ocr()
        print("OCR pre-initialized successfully")
    except Exception as e:
        print(f"OCR initialization error: {e}")
        traceback.print_exc()

# Global utility keywords for bill detection (weighted by confidence)
UTILITY_KEYWORDS = {
    # Universal utility terms
    'airtime': 4, 'data': 3, 'electricity': 4, 'water': 3, 'gas': 3,
    'internet': 3, 'broadband': 3, 'wifi': 3, 'phone': 2, 'mobile': 2,
    'recharge': 3, 'top up': 3, 'topup': 3, 'top-up': 3,
    'kwh': 4, 'meter': 3, 'units': 3, 'token': 3, 'vend': 3,
    'prepaid': 3, 'postpaid': 3, 'bill': 2, 'utility': 3,
    'successful': 2, 'transaction': 2, 'receipt': 2, 'paid': 2, 'payment': 2,

    # Nigerian telcos
    'glo': 3, 'mtn': 3, 'airtel': 3, '9mobile': 3, 'etisalat': 3,

    # Global telcos
    'vodafone': 3, 'orange': 3, 'safaricom': 3, 't-mobile': 3, 'verizon': 3,
    'at&t': 3, 'sprint': 3, 'ee': 3, 'three': 2, 'o2': 3, 'jio': 3,
    'aircel': 3, 'idea': 3, 'bsnl': 3, 'tigo': 3, 'movistar': 3,
    'claro': 3, 'telcel': 3, 'digicel': 3, 'econet': 3, 'telecel': 3,

    # Electricity companies (global)
    'eko': 3, 'ikeja': 3, 'ibedc': 3, 'aedc': 3, 'phed': 3, 'eedc': 3,  # Nigeria
    'eskom': 3, 'city power': 3,  # South Africa
    'ecg': 3, 'nedco': 3,  # Ghana
    'kplc': 3, 'kenya power': 3,  # Kenya
    'tanesco': 3,  # Tanzania
    'umeme': 3,  # Uganda
    'zesco': 3,  # Zambia
    'enel': 3, 'iberdrola': 3, 'edf': 3,  # Europe
    'pge': 3, 'duke energy': 3, 'con edison': 3,  # USA

    # Cable/TV
    'cable': 3, 'tv': 2, 'dstv': 3, 'gotv': 3, 'startimes': 3, 'showmax': 3,
    'netflix': 3, 'hulu': 3, 'disney': 2, 'hbo': 2, 'prime video': 2,
    'multichoice': 3, 'sky': 2, 'directv': 3,

    # Generic payment terms
    'confirmed': 2, 'approved': 2, 'completed': 2, 'success': 2,
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
    bill_category: str  # 'utility', 'transfer', 'deposit', 'unknown'
    transaction_id: str | None
    date: str | None
    time: str | None
    utility_score: int
    confidence: float
    supported_currencies: list[str] = []  # List of detected currencies
    # Validation fields
    is_valid_bill: bool = False  # Whether this is a valid utility bill
    rejection_reason: str | None = None  # Reason if rejected
    receipt_date_iso: str | None = None  # Parsed date in ISO format
    provider: str | None = None  # Detected provider (OPay, PalmPay, etc.)


def preprocess_image(img_bytes: bytes, aggressive: bool = False) -> np.ndarray:
    """Preprocess image for better OCR

    Args:
        img_bytes: Raw image bytes
        aggressive: If True, apply more aggressive preprocessing (use as fallback)
    """
    # Load image
    img = Image.open(io.BytesIO(img_bytes))

    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Keep images larger for better OCR (don't resize too small)
    max_dim = 2000
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    elif max(img.size) < 800:
        # Upscale small images for better OCR
        ratio = 800 / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    # Convert to numpy
    img_np = np.array(img)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    if aggressive:
        # Only apply aggressive preprocessing as fallback
        # Light denoise (less aggressive)
        img_cv = cv2.fastNlMeansDenoisingColored(img_cv, None, 3, 3, 7, 21)

        # Enhance contrast
        lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        img_cv = cv2.merge([l, a, b])
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_LAB2BGR)

    return img_cv


def extract_text(img_cv: np.ndarray, min_confidence: float = 0.3) -> tuple[str, float]:
    """Extract text using PaddleOCR

    Args:
        img_cv: OpenCV image in BGR format
        min_confidence: Minimum confidence threshold for text lines
    """
    ocr_instance = get_ocr()

    try:
        # Call OCR
        result = ocr_instance.ocr(img_cv)
        print(f"OCR result type: {type(result)}, has content: {bool(result)}")

        if not result or not result[0]:
            print("OCR returned empty result")
            return "", 0.0

        lines = []
        confidences = []

        for line in result[0]:
            if line and len(line) >= 2:
                text = line[1][0]
                conf = line[1][1]
                # Lower confidence threshold to capture more text
                if conf > min_confidence:
                    lines.append(str(text))
                    confidences.append(conf)

        full_text = '\n'.join(lines)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0
        print(f"Extracted {len(lines)} lines, {len(full_text)} chars, avg confidence: {avg_conf:.2f}")

        return full_text, avg_conf

    except Exception as e:
        print(f"OCR extraction error: {e}")
        traceback.print_exc()
        raise


def preprocess_grayscale_binary(img_bytes: bytes) -> np.ndarray:
    """Preprocess with grayscale and adaptive thresholding for difficult images"""
    img = Image.open(io.BytesIO(img_bytes))

    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Keep larger size for OCR
    max_dim = 2000
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    elif max(img.size) < 1000:
        ratio = 1000 / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    img_np = np.array(img)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding for better text extraction
    # This works well for receipts with varying backgrounds
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Convert back to BGR for PaddleOCR
    img_processed = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    return img_processed


def preprocess_sharpen(img_bytes: bytes) -> np.ndarray:
    """Preprocess with sharpening for blurry images"""
    img = Image.open(io.BytesIO(img_bytes))

    if img.mode != 'RGB':
        img = img.convert('RGB')

    max_dim = 2000
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    elif max(img.size) < 1000:
        ratio = 1000 / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    img_np = np.array(img)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Apply sharpening kernel
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    img_sharp = cv2.filter2D(img_cv, -1, kernel)

    # Reduce noise slightly
    img_sharp = cv2.bilateralFilter(img_sharp, 9, 75, 75)

    return img_sharp


def extract_text_with_retry(img_bytes: bytes) -> tuple[str, float, np.ndarray]:
    """Extract text with retry using different preprocessing strategies

    Returns: (text, confidence, processed_image)
    """
    best_text = ""
    best_conf = 0.0
    best_img = None

    # Strategy 1: Minimal preprocessing (best for clean screenshots)
    print("OCR Strategy 1: Minimal preprocessing...")
    img_cv = preprocess_image(img_bytes, aggressive=False)
    text1, conf1 = extract_text(img_cv, min_confidence=0.3)

    if len(text1) > len(best_text):
        best_text, best_conf, best_img = text1, conf1, img_cv

    # If we got good results, return them
    if len(text1) > 100 and conf1 > 0.7:
        print(f"Strategy 1 succeeded: {len(text1)} chars, {conf1:.2f} confidence")
        return text1, conf1, img_cv

    # Strategy 2: CLAHE contrast enhancement
    print("OCR Strategy 2: Contrast enhancement...")
    img_cv_aggressive = preprocess_image(img_bytes, aggressive=True)
    text2, conf2 = extract_text(img_cv_aggressive, min_confidence=0.25)

    if len(text2) > len(best_text):
        best_text, best_conf, best_img = text2, conf2, img_cv_aggressive

    if len(text2) > 100 and conf2 > 0.7:
        print(f"Strategy 2 succeeded: {len(text2)} chars, {conf2:.2f} confidence")
        return text2, conf2, img_cv_aggressive

    # Strategy 3: Grayscale with adaptive thresholding (good for colored backgrounds)
    print("OCR Strategy 3: Grayscale + binarization...")
    try:
        img_cv_binary = preprocess_grayscale_binary(img_bytes)
        text3, conf3 = extract_text(img_cv_binary, min_confidence=0.2)

        if len(text3) > len(best_text):
            best_text, best_conf, best_img = text3, conf3, img_cv_binary

        if len(text3) > 100:
            print(f"Strategy 3 succeeded: {len(text3)} chars, {conf3:.2f} confidence")
            return text3, conf3, img_cv_binary
    except Exception as e:
        print(f"Strategy 3 failed: {e}")

    # Strategy 4: Sharpening (good for blurry images)
    print("OCR Strategy 4: Sharpening...")
    try:
        img_cv_sharp = preprocess_sharpen(img_bytes)
        text4, conf4 = extract_text(img_cv_sharp, min_confidence=0.2)

        if len(text4) > len(best_text):
            best_text, best_conf, best_img = text4, conf4, img_cv_sharp

        if len(text4) > 100:
            print(f"Strategy 4 succeeded: {len(text4)} chars, {conf4:.2f} confidence")
            return text4, conf4, img_cv_sharp
    except Exception as e:
        print(f"Strategy 4 failed: {e}")

    # Return the best result we got
    print(f"Best result: {len(best_text)} chars, {best_conf:.2f} confidence")
    return best_text, best_conf, best_img if best_img is not None else img_cv


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

    # Ambiguous symbols that need word boundaries to avoid false positives
    AMBIGUOUS_SYMBOLS = {'DA', 'R', 'E', 'K', 'S'}

    # Symbol-based patterns (high priority)
    for symbol, code in CURRENCY_SYMBOLS.items():
        if code not in patterns:
            patterns[code] = []
        # Escape special regex chars in symbol
        escaped = re.escape(symbol)

        # Use word boundaries for short/ambiguous symbols
        if symbol in AMBIGUOUS_SYMBOLS:
            patterns[code].append(rf'\b{escaped}\s*([\d,]+(?:\.\d{{1,2}})?)')
            patterns[code].append(rf'([\d,]+(?:\.\d{{1,2}})?)\s*{escaped}\b')
        else:
            patterns[code].append(rf'{escaped}\s*([\d,]+(?:\.\d{{1,2}})?)')
            patterns[code].append(rf'([\d,]+(?:\.\d{{1,2}})?)\s*{escaped}')

    # Code-based patterns - always use word boundaries to avoid false positives
    for code in CURRENCY_CODES:
        if code not in patterns:
            patterns[code] = []
        # Use word boundaries for ALL currency codes to prevent matching within garbage text
        patterns[code].append(rf'\b{code}\b\s*([\d,]+(?:\.\d{{1,2}})?)')
        patterns[code].append(rf'([\d,]+(?:\.\d{{1,2}})?)\s*\b{code}\b')

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


def detect_region_from_text(text: str) -> str | None:
    """Detect likely region from provider names in text"""
    text_lower = text.lower()

    # Nigerian providers
    nigerian_providers = ['opay', 'palmpay', 'moniepoint', 'kuda', 'firstbank', 'gtbank',
                          'access bank', 'zenith', 'uba', 'vbank', 'ecobank', 'polaris',
                          'wema', 'fcmb', 'fidelity', 'stanbic', 'sterling', 'union bank',
                          'vulte', 'carbon', 'fairmoney', 'mtn', 'glo', 'airtel', '9mobile',
                          'eko disco', 'ikeja', 'ibedc', 'aedc', 'phed', 'dstv', 'gotv']
    if any(p in text_lower for p in nigerian_providers):
        return 'NG'

    # Kenyan providers
    kenyan_providers = ['m-pesa', 'mpesa', 'safaricom', 'equity bank', 'kcb', 'kplc', 'kenya power']
    if any(p in text_lower for p in kenyan_providers):
        return 'KE'

    # South African providers
    sa_providers = ['fnb', 'standard bank', 'nedbank', 'absa', 'capitec', 'vodacom', 'eskom']
    if any(p in text_lower for p in sa_providers):
        return 'ZA'

    # Ghanaian providers
    gh_providers = ['mtn ghana', 'vodafone ghana', 'airteltigo', 'ecg', 'ghana']
    if any(p in text_lower for p in gh_providers):
        return 'GH'

    return None


def extract_amount_with_currency(text: str) -> tuple[float, str]:
    """Extract monetary amount and detect currency from any global currency"""
    results = {}  # currency -> list of (amount, confidence_score)

    # Detect region to prioritize local currency
    region = detect_region_from_text(text)
    region_currency = {
        'NG': 'NGN', 'KE': 'KES', 'ZA': 'ZAR', 'GH': 'GHS',
        'US': 'USD', 'GB': 'GBP', 'EU': 'EUR'
    }.get(region)

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
        # First, try to find amount with keyword context
        generic_pattern = r'(?:total|amount|paid|sum|balance|successful)[:\s]*([\d,]+(?:\.\d{1,2})?)'
        matches = re.findall(generic_pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                amount = float(match.replace(',', ''))
                if 1 <= amount <= 100000000:
                    # Use region currency if detected, otherwise UNKNOWN
                    return amount, region_currency or 'UNKNOWN'
            except ValueError:
                continue

        # If we have a region, try finding any reasonable amount
        if region_currency:
            amount_pattern = r'([\d,]+\.\d{2}|\d{1,3}(?:,\d{3})+(?:\.\d{2})?|\d{4,}(?:\.\d{2})?)'
            matches = re.findall(amount_pattern, text)
            valid_amounts = []
            for match in matches:
                try:
                    amount = float(match.replace(',', ''))
                    if region_currency == 'NGN' and 50 <= amount <= 10000000:
                        valid_amounts.append(amount)
                    elif 1 <= amount <= 100000000:
                        valid_amounts.append(amount)
                except ValueError:
                    continue

            if valid_amounts:
                best_amount = max(valid_amounts)
                print(f"No currency patterns matched, using region fallback: {best_amount} {region_currency}")
                return best_amount, region_currency

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

    # If we detected a region, prioritize that currency first
    if region_currency and region_currency in results:
        top_match = results[region_currency][0]
        best_currency = region_currency
        best_amount = top_match[0]
        best_confidence = top_match[1] + 5  # Boost confidence for region match
        print(f"Region-based currency priority: {region} -> {region_currency}, amount={best_amount}")

    for currency in priority_order:
        if currency in results:
            top_match = results[currency][0]
            # For region currency, we already handled it with boost
            if currency == region_currency:
                continue
            if top_match[1] > best_confidence or (top_match[1] == best_confidence and top_match[0] > best_amount):
                best_currency = currency
                best_amount = top_match[0]
                best_confidence = top_match[1]

    # If none from priority, use first found
    if not best_currency and results:
        best_currency = list(results.keys())[0]
        best_amount = results[best_currency][0][0]

    # Final fallback: if we detected a region but no good amount, try generic extraction
    if region_currency:
        # If best amount seems wrong (too small for region), override with generic extraction
        should_try_generic = (
            not best_currency or
            (region_currency == 'NGN' and best_amount < 50) or
            (region_currency != 'NGN' and best_amount < 1)
        )

        if should_try_generic:
            # Look for amounts that look like bill payments (e.g., 11000.00, 5,000, 200.00)
            generic_pattern = r'([\d,]+\.\d{2}|\d{1,3}(?:,\d{3})+|\d{3,})'
            matches = re.findall(generic_pattern, text)

            valid_amounts = []
            for match in matches:
                try:
                    amount = float(match.replace(',', ''))
                    # For NGN, typical amounts are 50-10000000
                    if region_currency == 'NGN' and 50 <= amount <= 10000000:
                        valid_amounts.append(amount)
                    elif region_currency != 'NGN' and 1 <= amount <= 100000000:
                        valid_amounts.append(amount)
                except ValueError:
                    continue

            if valid_amounts:
                # Take the largest reasonable amount (likely the transaction amount)
                best_generic = max(valid_amounts)
                print(f"Generic extraction with region currency: {best_generic} {region_currency} (overriding {best_amount} {best_currency})")
                return best_generic, region_currency

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


def detect_bill_type(text: str) -> tuple[str, str]:
    """Detect type of bill and category (GLOBAL support)
    Returns: (bill_type, bill_category)
    Categories: 'utility', 'transfer', 'deposit', 'unknown'
    """
    text_lower = text.lower()

    # ===== AIRTIME / MOBILE TOP-UP (Global) =====
    airtime_keywords = [
        'airtime', 'recharge', 'top up', 'topup', 'top-up', 'mobile credit',
        'phone credit', 'call credit', 'prepaid credit', 'recarga', 'recarrega',
        'aufladung', 'ricarica', 'пополнение',  # Spanish, Portuguese, German, Italian, Russian
    ]
    # Telco names that indicate airtime
    telco_names = [
        'mtn', 'glo', 'airtel', '9mobile', 'etisalat',  # Nigeria
        'safaricom', 'airtel kenya', 'telkom kenya',  # Kenya
        'vodacom', 'cell c', 'telkom',  # South Africa
        'vodafone', 'mtn ghana', 'airteltigo',  # Ghana
        'orange', 'moov', 'togocel',  # West Africa
        't-mobile', 'verizon', 'at&t', 'sprint',  # USA
        'ee', 'o2', 'three', 'vodafone uk',  # UK
        'jio', 'vi', 'bsnl', 'airtel india',  # India
        'claro', 'movistar', 'telcel',  # Latin America
        'digicel',  # Caribbean
    ]
    if any(kw in text_lower for kw in airtime_keywords):
        return 'airtime', 'utility'
    if any(telco in text_lower for telco in telco_names) and any(kw in text_lower for kw in ['successful', 'completed', 'paid', 'receipt', 'transaction']):
        return 'airtime', 'utility'

    # ===== DATA BUNDLE (Global) =====
    data_keywords = ['data', 'bundle', 'internet pack', 'data plan', 'mb', 'gb', 'datos', 'données']
    data_context = ['plan', 'package', 'bundle', 'subscription', 'validity', 'valid for']
    if any(kw in text_lower for kw in data_keywords) and any(ctx in text_lower for ctx in data_context + telco_names):
        return 'data', 'utility'

    # ===== ELECTRICITY (Global) =====
    electricity_keywords = [
        'electricity', 'electric', 'power', 'kwh', 'kilowatt', 'meter', 'energia',
        'électricité', 'strom', 'elettricità', 'электричество',
        'prepaid meter', 'postpaid', 'token', 'units', 'vend',
    ]
    electricity_providers = [
        # Nigeria
        'eko disco', 'ikeja', 'ibedc', 'aedc', 'phed', 'eedc', 'kedco', 'kaedco', 'bedc', 'jedc',
        # South Africa
        'eskom', 'city power', 'cape town electricity',
        # Ghana
        'ecg', 'nedco', 'ghana grid',
        # Kenya
        'kplc', 'kenya power',
        # Other Africa
        'tanesco', 'umeme', 'zesco', 'escom', 'eswatini electricity',
        # International
        'enel', 'iberdrola', 'edf', 'engie', 'e.on', 'vattenfall',
        'pge', 'duke energy', 'con edison', 'florida power', 'southern company',
    ]
    if any(kw in text_lower for kw in electricity_keywords):
        return 'electricity', 'utility'
    if any(provider in text_lower for provider in electricity_providers):
        return 'electricity', 'utility'

    # ===== WATER (Global) =====
    water_keywords = [
        'water', 'água', 'eau', 'wasser', 'acqua', 'вода',
        'water bill', 'water utility', 'water board', 'water corp',
    ]
    water_providers = ['fctwb', 'lagos water', 'rand water', 'joburg water', 'cape town water']
    if any(kw in text_lower for kw in water_keywords):
        return 'water', 'utility'
    if any(provider in text_lower for provider in water_providers):
        return 'water', 'utility'

    # ===== INTERNET / BROADBAND (Global) =====
    internet_keywords = [
        'internet', 'wifi', 'broadband', 'fiber', 'fibre', 'dsl', 'router',
        'isp', 'home internet', 'business internet',
    ]
    internet_providers = [
        'spectranet', 'smile', 'swift', 'ipnx', 'coollink',  # Nigeria
        'comcast', 'xfinity', 'spectrum', 'cox',  # USA
        'bt', 'virgin media', 'sky broadband', 'talktalk',  # UK
        'telkom', 'mweb', 'afrihost',  # South Africa
    ]
    if any(kw in text_lower for kw in internet_keywords):
        return 'internet', 'utility'
    if any(provider in text_lower for provider in internet_providers):
        return 'internet', 'utility'

    # ===== GAS (Global) =====
    gas_keywords = ['gas', 'natural gas', 'cooking gas', 'lpg', 'propane', 'butane', 'gaz']
    if any(kw in text_lower for kw in gas_keywords) and not 'vegas' in text_lower:
        return 'gas', 'utility'

    # ===== CABLE / TV / STREAMING (Global) =====
    cable_keywords = [
        'cable', 'satellite', 'tv subscription', 'television',
        'dstv', 'gotv', 'startimes', 'showmax',  # Africa
        'netflix', 'hulu', 'disney+', 'hbo', 'prime video', 'youtube premium',
        'sky', 'directv', 'dish network', 'sling',  # International
        'multichoice',
    ]
    if any(kw in text_lower for kw in cable_keywords):
        return 'cable', 'utility'

    # ===== BETTING / GAMING (Global) =====
    betting_keywords = [
        'bet9ja', 'sportybet', 'betway', 'nairabet', '1xbet', 'betking',
        'fanduel', 'draftkings', 'betmgm', 'caesars', 'bet365',
        'paddy power', 'william hill', 'ladbrokes', 'betfair',
    ]
    if any(kw in text_lower for kw in betting_keywords):
        return 'betting', 'utility'

    # ===== Check for TRANSFERS (NOT VALID utility bills) =====
    transfer_keywords = [
        'transfer', 'bank transfer', 'inter-bank', 'interbank', 'nip transfer',
        'wire transfer', 'money transfer', 'funds transfer',
        'beneficiary', 'sender', 'recipient account', 'destination bank',
        'narration', 'other local banks', 'swift', 'iban',
    ]
    # But check if it's actually a utility payment via bank transfer first
    utility_via_transfer = [
        'vbank|glo', 'vbank|mtn', 'vbank|airtel', 'chamswitch|',
        'airtime|', 'data|', 'electricity|', 'dstv|', 'gotv|',
    ]
    if any(kw in text_lower for kw in transfer_keywords):
        if any(util in text_lower for util in utility_via_transfer):
            if 'data|' in text_lower or 'data bundle' in text_lower:
                return 'data', 'utility'
            elif 'electricity' in text_lower or 'disco' in text_lower:
                return 'electricity', 'utility'
            elif 'dstv' in text_lower or 'gotv' in text_lower:
                return 'cable', 'utility'
            else:
                return 'airtime', 'utility'
        return 'transfer', 'transfer'

    # ===== Check for DEPOSITS (NOT VALID) =====
    deposit_keywords = [
        'deposit', 'bank deposit', 'credited to', 'credit alert',
        'inward transfer', 'incoming transfer', 'received from',
    ]
    if any(kw in text_lower for kw in deposit_keywords):
        return 'deposit', 'deposit'

    # Unknown - needs manual review
    return 'unknown', 'unknown'


def detect_provider(text: str) -> str | None:
    """Detect the payment provider from the receipt (GLOBAL support)"""
    text_lower = text.lower()

    providers = {
        # Nigerian Banks & Fintechs
        'opay': ['opay', 'o-pay'],
        'palmpay': ['palmpay', 'palm pay'],
        'moniepoint': ['moniepoint', 'monie point'],
        'kuda': ['kuda'],
        'firstbank': ['firstbank', 'first bank', 'firstmobile'],
        'gtbank': ['gtbank', 'gt bank', 'guaranty trust'],
        'accessbank': ['access bank', 'accessbank', 'accessmore'],
        'zenithbank': ['zenith bank', 'zenithbank'],
        'ubabank': ['uba', 'united bank for africa'],
        'vbank': ['vbank', 'vfd', 'v bank'],
        'ecobank': ['ecobank', 'eco bank'],
        'polaris': ['polaris', 'polaris bank'],
        'wema': ['wema', 'alat'],
        'fcmb': ['fcmb', 'first city'],
        'fidelity': ['fidelity'],
        'stanbic': ['stanbic', 'stanbic ibtc'],
        'sterling': ['sterling'],
        'unionbank': ['union bank'],
        'vulte': ['vulte', 'vul-te'],
        'carbon': ['carbon', 'paylater'],
        'fairmoney': ['fairmoney', 'fair money'],

        # Nigerian Telcos
        'glo': ['glo', 'globacom'],
        'mtn_ng': ['mtn nigeria', 'mtn ng'],
        'airtel_ng': ['airtel nigeria'],
        '9mobile': ['9mobile', 'etisalat'],

        # Kenyan
        'mpesa': ['m-pesa', 'mpesa', 'safaricom'],
        'equitybank': ['equity bank', 'equity mobile'],
        'kcb': ['kcb', 'kenya commercial'],

        # South African
        'fnb': ['fnb', 'first national bank'],
        'standardbank': ['standard bank'],
        'nedbank': ['nedbank'],
        'absa': ['absa', 'barclays'],
        'capitec': ['capitec'],

        # Ghanaian
        'mtn_gh': ['mtn ghana'],
        'vodafone_gh': ['vodafone ghana'],
        'airteltigo': ['airteltigo'],

        # Global Mobile Money
        'mtn_momo': ['mtn mobile money', 'momo', 'mobile money'],
        'orange_money': ['orange money'],

        # International Banks
        'chase': ['chase', 'jpmorgan'],
        'bofa': ['bank of america', 'bofa'],
        'wells_fargo': ['wells fargo'],
        'citibank': ['citibank', 'citi'],
        'hsbc': ['hsbc'],
        'barclays_uk': ['barclays'],
        'lloyds': ['lloyds'],
        'natwest': ['natwest'],

        # Global Fintechs
        'paypal': ['paypal'],
        'venmo': ['venmo'],
        'cashapp': ['cash app', 'cashapp', 'square cash'],
        'revolut': ['revolut'],
        'wise': ['wise', 'transferwise'],
        'payoneer': ['payoneer'],
        'skrill': ['skrill'],
        'neteller': ['neteller'],

        # Payment Processors
        'stripe': ['stripe'],
        'flutterwave': ['flutterwave'],
        'paystack': ['paystack'],
        'interswitch': ['interswitch', 'quickteller'],
    }

    for provider, keywords in providers.items():
        if any(kw in text_lower for kw in keywords):
            return provider

    return None


def parse_receipt_date(date_str: str | None, time_str: str | None = None) -> str | None:
    """Parse receipt date string into ISO format"""
    if not date_str:
        return None

    from datetime import datetime
    import calendar

    # Clean up the date string
    date_str = date_str.strip()

    # Common date formats to try
    formats = [
        '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d',
        '%d-%m-%Y', '%m-%d-%Y', '%Y-%m-%d',
        '%d %B %Y', '%d %b %Y', '%B %d, %Y', '%b %d, %Y',
        '%d %B, %Y', '%d %b, %Y',
        '%Y-%m-%d %H:%M:%S',
        '%d/%m/%y', '%m/%d/%y',
    ]

    # Handle formats like "Mar 17th, 2025" or "March 17, 2025"
    # Remove ordinal suffixes
    import re
    date_str_clean = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str)

    for fmt in formats:
        try:
            dt = datetime.strptime(date_str_clean, fmt)
            # Add time if available
            if time_str:
                try:
                    time_str_clean = time_str.strip().upper()
                    for tfmt in ['%H:%M:%S', '%H:%M', '%I:%M:%S %p', '%I:%M %p', '%I:%M:%S%p', '%I:%M%p']:
                        try:
                            time_part = datetime.strptime(time_str_clean, tfmt)
                            dt = dt.replace(hour=time_part.hour, minute=time_part.minute, second=time_part.second)
                            break
                        except ValueError:
                            continue
                except:
                    pass
            return dt.isoformat()
        except ValueError:
            continue

    return None


def validate_bill(bill_type: str, bill_category: str, amount: float,
                  receipt_date_iso: str | None, utility_score: int,
                  currency: str = 'NGN', amount_usd: float = 0) -> tuple[bool, str | None]:
    """
    Validate if a bill should be approved or rejected (GLOBAL support).
    Returns: (is_valid, rejection_reason)
    """
    from datetime import datetime, timedelta

    # Rule 1: Must be a utility bill category
    if bill_category not in ['utility']:
        if bill_category == 'transfer':
            return False, "Bank transfers are not eligible. Only utility bill payments (airtime, data, electricity, etc.) are accepted."
        elif bill_category == 'deposit':
            return False, "Bank deposits are not eligible. Only utility bill payments are accepted."
        else:
            return False, "Unable to identify bill type. Please upload a clear utility bill receipt."

    # Rule 2: Must have a valid amount
    if amount <= 0:
        return False, "Could not detect bill amount. Please ensure the amount is clearly visible."

    # Rule 3: Minimum amount threshold (currency-aware)
    # Use USD equivalent for global comparison (~$0.03 minimum)
    MIN_AMOUNT_USD = 0.03
    actual_usd = amount_usd if amount_usd > 0 else amount * 0.000625  # Fallback to NGN rate

    if actual_usd < MIN_AMOUNT_USD:
        return False, f"Bill amount is below minimum threshold (equivalent to ~${MIN_AMOUNT_USD:.2f} USD)."

    # Rule 4: Check receipt age (must be within 24 hours)
    if receipt_date_iso:
        try:
            receipt_dt = datetime.fromisoformat(receipt_date_iso)
            now = datetime.now()
            age = now - receipt_dt

            # Allow up to 24 hours
            if age > timedelta(hours=24):
                hours_old = age.total_seconds() / 3600
                return False, f"Receipt is {hours_old:.0f} hours old. Only receipts from the last 24 hours are accepted."

            # Check if receipt is from the future (suspicious)
            if receipt_dt > now + timedelta(hours=1):  # Allow 1 hour tolerance
                return False, "Receipt date appears to be in the future. Please upload a valid receipt."
        except:
            pass  # If we can't parse the date, we'll allow it for manual review

    # Rule 5: Utility score threshold
    # If we confidently identified a specific utility type, lower the threshold
    KNOWN_UTILITY_TYPES = ['airtime', 'data', 'electricity', 'water', 'gas', 'internet', 'cable', 'betting']
    if bill_type in KNOWN_UTILITY_TYPES:
        # We identified a specific utility type, be more lenient
        MIN_UTILITY_SCORE = 2
    else:
        # Unknown type, require higher confidence
        MIN_UTILITY_SCORE = 4

    if utility_score < MIN_UTILITY_SCORE:
        return False, f"Receipt does not appear to be a valid utility bill (confidence too low)."

    # All checks passed
    return True, None


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


@app.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "ok", "ocr_ready": ocr_ready}


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
            print(f"Received image: {len(img_bytes)} bytes")
        except Exception as e:
            print(f"Base64 decode error: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")

        # Extract text with retry strategies
        print("Starting OCR extraction...")
        text, confidence, img_cv = extract_text_with_retry(img_bytes)
        print(f"Final extraction: {len(text)} chars, confidence: {confidence:.2f}")

        if not text:
            return OCRResponse(
                success=False,
                text="",
                amount=0,
                currency="UNKNOWN",
                amount_usd=0,
                bill_type="unknown",
                bill_category="unknown",
                transaction_id=None,
                date=None,
                time=None,
                utility_score=0,
                confidence=0,
                supported_currencies=CURRENCY_CODES[:20],
                is_valid_bill=False,
                rejection_reason="No text could be extracted from the image.",
                receipt_date_iso=None,
                provider=None
            )

        # Extract fields
        amount, currency = extract_amount_with_currency(text)
        amount_usd = convert_to_usd(amount, currency)
        transaction_id = extract_transaction_id(text)
        date, time = extract_date_time(text)
        bill_type, bill_category = detect_bill_type(text)
        utility_score = calculate_utility_score(text)
        provider = detect_provider(text)
        receipt_date_iso = parse_receipt_date(date, time)

        # Validate the bill (with currency-aware thresholds)
        is_valid_bill, rejection_reason = validate_bill(
            bill_type, bill_category, amount, receipt_date_iso, utility_score,
            currency=currency, amount_usd=amount_usd
        )

        # Debug: Log first 500 chars of extracted text to help debug OCR issues
        print(f"OCR Text Preview: {text[:500] if len(text) > 500 else text}")
        print(f"Extracted: amount={amount} {currency}, type={bill_type}, category={bill_category}, provider={provider}")
        print(f"Validation: valid={is_valid_bill}, reason={rejection_reason}")

        # If provider is Nigerian but currency isn't NGN, might be a detection issue
        nigerian_providers = ['opay', 'palmpay', 'moniepoint', 'kuda', 'firstbank', 'gtbank',
                              'accessbank', 'zenithbank', 'ubabank', 'vbank', 'ecobank']
        if provider and provider.lower() in nigerian_providers and currency not in ['NGN', 'UNKNOWN']:
            print(f"Warning: Nigerian provider {provider} detected but currency is {currency} - may be a false positive")

        return OCRResponse(
            success=True,
            text=text,
            amount=amount,
            currency=currency,
            amount_usd=round(amount_usd, 4),
            bill_type=bill_type,
            bill_category=bill_category,
            transaction_id=transaction_id,
            date=date,
            time=time,
            utility_score=utility_score,
            confidence=confidence,
            supported_currencies=CURRENCY_CODES[:20],
            is_valid_bill=is_valid_bill,
            rejection_reason=rejection_reason,
            receipt_date_iso=receipt_date_iso,
            provider=provider
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"OCR processing error: {e}")
        traceback.print_exc()
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
