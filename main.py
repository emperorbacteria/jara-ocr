"""
Jara OCR Service - Gemini Vision Edition
Uses Google's Gemini 1.5 Flash for intelligent receipt analysis
"""

import os
import json
import base64
from datetime import datetime, timedelta
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import google.generativeai as genai

# Initialize FastAPI
app = FastAPI(title="Jara OCR Service", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    print("Gemini API configured successfully")
else:
    print("WARNING: GOOGLE_API_KEY not set!")

# Response model
class OCRResponse(BaseModel):
    success: bool
    text: str
    amount: float
    currency: str
    amount_usd: float
    bill_type: str
    bill_category: str
    transaction_id: Optional[str]
    date: Optional[str]
    time: Optional[str]
    utility_score: int
    confidence: float
    is_valid_bill: bool
    rejection_reason: Optional[str]
    receipt_date_iso: Optional[str]
    provider: Optional[str]
    error: Optional[str] = None

# Exchange rates (USD base) - update periodically
EXCHANGE_RATES = {
    'USD': 1.0,
    'NGN': 0.00063,   # 1 NGN = 0.00063 USD (approx 1590 NGN/USD)
    'GBP': 1.27,
    'EUR': 1.09,
    'KES': 0.0077,    # Kenya Shilling
    'GHS': 0.064,     # Ghana Cedi
    'ZAR': 0.053,     # South African Rand
    'INR': 0.012,
    'CAD': 0.74,
    'AUD': 0.65,
}

def convert_to_usd(amount: float, currency: str) -> float:
    """Convert amount to USD"""
    if amount <= 0:
        return 0
    rate = EXCHANGE_RATES.get(currency, EXCHANGE_RATES.get('USD', 1.0))
    return amount * rate

# The prompt for Gemini to analyze receipts
RECEIPT_ANALYSIS_PROMPT = """Analyze this receipt/payment screenshot image and extract the following information.

IMPORTANT CLASSIFICATION RULES:
1. UTILITY BILLS (category: "utility") - ONLY these are valid:
   - Airtime/phone credit top-up
   - Mobile data purchase
   - Electricity bill payment
   - Water bill payment
   - Internet/WiFi subscription
   - Cable TV (DSTV, GoTV, Startimes)
   - Gas bill payment
   - Betting/gaming top-up

2. BANK TRANSFERS (category: "transfer") - NOT valid:
   - Money sent to another person
   - Transfer to bank account
   - Payment to individual names
   - P2P transfers

3. DEPOSITS (category: "deposit") - NOT valid:
   - Money received/credited
   - Incoming transfers
   - Bank deposits

4. If you cannot clearly determine the type, use category: "needs_review"

Return a JSON object with these exact fields:
{
  "transaction_type": "utility" | "transfer" | "deposit" | "unknown",
  "bill_type": "airtime" | "data" | "electricity" | "water" | "internet" | "cable" | "gas" | "betting" | "transfer" | "deposit" | "unknown",
  "amount": <number - the transaction amount, 0 if not found>,
  "currency": "<3-letter ISO code like NGN, USD, GBP, EUR, KES, GHS, ZAR>",
  "provider": "<payment provider name like OPay, PalmPay, Moniepoint, MTN, etc.>",
  "transaction_id": "<reference number or transaction ID if visible>",
  "date": "<date in format DD/MM/YYYY or as shown>",
  "time": "<time if visible>",
  "recipient": "<who received the payment - company name for utility, person name for transfer>",
  "confidence": <0.0-1.0 how confident you are in this analysis>,
  "raw_text": "<all text you can read from the image>",
  "reasoning": "<brief explanation of why you classified it this way>"
}

Be very careful to distinguish between:
- Paying FOR airtime/data (utility - valid)
- Transferring money TO someone (transfer - not valid)

Look for keywords like:
- "Airtime purchase", "Data bundle", "Electricity token" = utility
- "Transfer to", "Sent to", "Payment to [person name]" = transfer
- "Credited", "Received from" = deposit

Return ONLY the JSON object, no other text."""


async def analyze_with_gemini(image_bytes: bytes) -> dict:
    """Use Gemini Vision to analyze receipt image"""
    try:
        # Initialize model
        model = genai.GenerativeModel('gemini-1.5-flash')

        # Create image part
        image_part = {
            "mime_type": "image/jpeg",
            "data": base64.b64encode(image_bytes).decode('utf-8')
        }

        # Generate response
        response = model.generate_content(
            [RECEIPT_ANALYSIS_PROMPT, image_part],
            generation_config=genai.GenerationConfig(
                temperature=0.1,  # Low temperature for consistent extraction
                max_output_tokens=1024,
            )
        )

        # Parse JSON response
        response_text = response.text.strip()

        # Clean up response if it has markdown code blocks
        if response_text.startswith('```'):
            response_text = response_text.split('```')[1]
            if response_text.startswith('json'):
                response_text = response_text[4:]
        response_text = response_text.strip()

        result = json.loads(response_text)
        return result

    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"Response was: {response_text[:500] if 'response_text' in dir() else 'N/A'}")
        return {"error": f"Failed to parse Gemini response: {str(e)}"}
    except Exception as e:
        print(f"Gemini API error: {e}")
        return {"error": str(e)}


def validate_bill(
    bill_type: str,
    bill_category: str,
    amount: float,
    receipt_date_iso: Optional[str],
    currency: str,
    amount_usd: float
) -> tuple[bool, Optional[str]]:
    """Validate if the bill should be auto-approved"""

    # Rule 1: Must be a utility bill
    if bill_category == 'transfer':
        return False, "Bank transfers are not eligible. Only utility bill payments (airtime, data, electricity, etc.) are accepted."
    elif bill_category == 'deposit':
        return False, "Bank deposits are not eligible. Only utility bill payments are accepted."
    elif bill_category == 'needs_review':
        return False, None  # Needs manual review, no auto-reject
    elif bill_category not in ['utility']:
        return False, "Unable to identify bill type. Please upload a clear utility bill receipt."

    # Rule 2: Must have valid amount
    if amount <= 0:
        return False, "Could not detect bill amount. Please ensure the amount is clearly visible."

    # Rule 3: Minimum amount (currency-aware)
    min_amounts = {
        'NGN': 50,
        'USD': 0.50,
        'GBP': 0.50,
        'EUR': 0.50,
        'KES': 50,
        'GHS': 5,
        'ZAR': 10,
    }
    min_amount = min_amounts.get(currency, 0.50)
    if amount < min_amount:
        return False, f"Bill amount too small. Minimum is {currency} {min_amount}."

    # Rule 4: Receipt date check (if available)
    if receipt_date_iso:
        try:
            receipt_dt = datetime.fromisoformat(receipt_date_iso.replace('Z', '+00:00'))
            now = datetime.now(receipt_dt.tzinfo) if receipt_dt.tzinfo else datetime.now()

            # Check if receipt is too old (> 24 hours)
            if (now - receipt_dt) > timedelta(hours=24):
                return False, "Receipt is older than 24 hours. Please upload a recent receipt."

            # Check if receipt is in the future
            if receipt_dt > now + timedelta(hours=1):
                return False, "Receipt date appears to be in the future. Please upload a valid receipt."
        except:
            pass  # If we can't parse date, allow for manual review

    # All checks passed
    return True, None


def parse_receipt_date(date_str: Optional[str], time_str: Optional[str]) -> Optional[str]:
    """Try to parse date/time into ISO format"""
    if not date_str:
        return None

    date_formats = [
        "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d %B %Y", "%d %b %Y",
        "%m/%d/%Y", "%Y/%m/%d", "%d.%m.%Y"
    ]

    for fmt in date_formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            if time_str:
                time_formats = ["%H:%M:%S", "%H:%M", "%I:%M %p", "%I:%M:%S %p"]
                for tfmt in time_formats:
                    try:
                        t = datetime.strptime(time_str.strip(), tfmt)
                        dt = dt.replace(hour=t.hour, minute=t.minute, second=t.second)
                        break
                    except:
                        continue
            return dt.isoformat()
        except:
            continue

    return None


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "jara-ocr-gemini",
        "version": "2.0.0",
        "gemini_configured": bool(GOOGLE_API_KEY)
    }


@app.post("/ocr", response_model=OCRResponse)
async def process_receipt(file: UploadFile = File(...)):
    """Process receipt image using Gemini Vision"""

    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not configured")

    try:
        # Read image
        image_bytes = await file.read()
        print(f"Processing image: {file.filename}, size: {len(image_bytes)} bytes")

        # Analyze with Gemini
        result = await analyze_with_gemini(image_bytes)

        if "error" in result:
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
                is_valid_bill=False,
                rejection_reason=f"Analysis failed: {result['error']}",
                receipt_date_iso=None,
                provider=None,
                error=result['error']
            )

        # Extract data from Gemini response
        transaction_type = result.get('transaction_type', 'unknown')
        bill_type = result.get('bill_type', 'unknown')
        amount = float(result.get('amount', 0))
        currency = result.get('currency', 'NGN')
        provider = result.get('provider')
        transaction_id = result.get('transaction_id')
        date = result.get('date')
        time = result.get('time')
        confidence = float(result.get('confidence', 0))
        raw_text = result.get('raw_text', '')
        reasoning = result.get('reasoning', '')

        # Map transaction type to bill category
        if transaction_type == 'utility':
            bill_category = 'utility'
        elif transaction_type == 'transfer':
            bill_category = 'transfer'
        elif transaction_type == 'deposit':
            bill_category = 'deposit'
        else:
            bill_category = 'needs_review'

        # Calculate USD amount
        amount_usd = convert_to_usd(amount, currency)

        # Calculate utility score (0-10 based on confidence and type)
        if bill_category == 'utility' and bill_type in ['airtime', 'data', 'electricity', 'water', 'internet', 'cable', 'gas', 'betting']:
            utility_score = min(10, int(confidence * 10) + 2)
        elif bill_category == 'utility':
            utility_score = min(10, int(confidence * 8))
        else:
            utility_score = 0

        # Parse receipt date
        receipt_date_iso = parse_receipt_date(date, time)

        # Validate bill
        is_valid_bill, rejection_reason = validate_bill(
            bill_type, bill_category, amount, receipt_date_iso, currency, amount_usd
        )

        # Log for debugging
        print(f"Gemini Analysis: type={transaction_type}, bill_type={bill_type}, amount={amount} {currency}")
        print(f"Category: {bill_category}, Valid: {is_valid_bill}, Reason: {rejection_reason}")
        print(f"Reasoning: {reasoning}")

        return OCRResponse(
            success=True,
            text=raw_text,
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
            is_valid_bill=is_valid_bill,
            rejection_reason=rejection_reason,
            receipt_date_iso=receipt_date_iso,
            provider=provider,
        )

    except Exception as e:
        print(f"Error processing receipt: {e}")
        import traceback
        traceback.print_exc()
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
            is_valid_bill=False,
            rejection_reason=f"Processing error: {str(e)}",
            receipt_date_iso=None,
            provider=None,
            error=str(e)
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting Jara OCR Service (Gemini) on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
