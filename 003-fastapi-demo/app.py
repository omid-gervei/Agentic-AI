from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import requests
import uuid
import json
from datetime import datetime
import asyncio
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "gemma3:1b"  # یا هر مدل دیگری که دارید

# Store for processing results
processing_results = {}

# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info(" Starting Product Description API")
    yield
    # Shutdown
    logger.info(" Shutting down Product Description API")

# Create FastAPI app
app = FastAPI(
    title="Product Description Generator API",
    description="API برای تولید عنوان، توضیح و کلمات کلیدی محصول",
    version="1.0.0",
    lifespan=lifespan
)

# Pydantic models for request/response
class ProductRequest(BaseModel):
    text: str = Field(..., min_length=5, max_length=500, description="متن عنوان محصول")
    request_id: Optional[str] = Field(None, description="شناسه درخواست (اختیاری)")

class ProductResponse(BaseModel):
    request_id: str = Field(..., description="شناسه درخواست")
    status: str = Field(..., description="وضعیت پردازش")
    optimized_title: Optional[str] = Field(None, description="عنوان بهینه شده")
    short_description: Optional[str] = Field(None, description="توضیح کوتاه محصول")
    keywords: Optional[str] = Field(None, description="کلمات کلیدی")
    created_at: datetime = Field(..., description="زمان ایجاد")
    processing_time: Optional[float] = Field(None, description="زمان پردازش (ثانیه)")

class StatusResponse(BaseModel):
    request_id: str = Field(..., description="شناسه درخواست")
    status: str = Field(..., description="وضعیت پردازش")
    progress: float = Field(..., description="پیشرفت (۰ تا ۱)")
    estimated_time: Optional[float] = Field(None, description="زمان تخمینی باقی‌مانده (ثانیه)")

# Helper function to call Ollama
def call_ollama(messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 500) -> Optional[str]:
    """تماس با مدل Ollama"""
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": MODEL_NAME,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            },
            timeout=1200
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("message", {}).get("content", "").strip()
        else:
            logger.error(f"خطا در ارتباط با مدل: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"خطا در ارتباط با Ollama: {e}")
        return None

def extract_product_title(raw_title: str) -> Optional[str]:
    """مدل ۱: استخراج عنوان مناسب از عنوان خام"""
    
    system_prompt = """شما یک متخصص بهینه‌سازی عنوان محصولات هستید. وظیفه شما استخراج عنوان مناسب و استاندارد از عنوان‌های خام محصولات است.

قوانین استخراج عنوان:
۱. حذف اطلاعات اضافی مانند قیمت، سال تولید، کد محصول (مگر ضروری باشد)
۲. حفظ نام برند و ویژگی‌های اصلی محصول
۳. استفاده از ساختار استاندارد: [نام محصول] + [ویژگی اصلی] + [برند] (اختیاری)
۴. کوتاه و گویا بودن (حداکثر ۱۰ کلمه)
۵. استفاده از کلمات کلیدی مهم در ابتدای عنوان
۶. حذف کلمات تکراری و زائد

نمونه‌های ورودی و خروجی:

ورودی: "ماست کپفی میهن ۱.۵% چربی کم چرب بسته بندی ۵۰۰ گرمی تولید ۱۴۰۳"
خروجی: "ماست کم چرب میهن ۱.۵% چربی"

ورودی: "شامپو حجم دهنده پرپشت کلیر ضد ریزش مو بسته ۴۰۰ میلی لیتر"
خروجی: "شامپو ضد ریزش مو و حجم دهنده کلیر"

ورودی: "کفش ورزشی نایک ایرمکس مشکی سایز ۴۲ مدل ۲۰۲۴ اصل"
خروجی: "کفش ورزشی نایک ایرمکس مشکی"

ورودی: "قهوه ترک عربیکا درجه یک دم‌آوری سنتی بسته ۲۵۰ گرم"
خروجی: "قهوه ترک عربیکا درجه یک"

ورودی: "تلویزیون ال‌جی ۵۵ اینچ اولد اسمارت ۴K مدل ۲۰۲۳"
خروجی: "تلویزیون ال‌جی ۵۵ اینچ اولد ۴K"

حالا عنوان زیر را بر اساس قوانین فوق بهینه کن:"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": raw_title}
    ]
    
    return call_ollama(messages, temperature=0.3, max_tokens=100)

def generate_short_description(product_title: str, raw_title: str) -> Optional[str]:
    """مدل ۲: تولید توضیح کوتاه ۴۰۰ کاراکتری"""
    
    system_prompt = """شما یک متخصص تولید محتوای محصولات هستید. وظیفه شما نوشتن توضیح کوتاه و جذاب برای محصولات است.

الزامات توضیح کوتاه:
۱. طول دقیقاً بین ۳۵۰ تا ۴۵۰ کاراکتر (با فاصله)
۲. شروع جذاب و گیرا
۳. ذکر ۳-۴ ویژگی اصلی محصول
۴. اشاره به مزایای استفاده از محصول
۵. پایان با فراخوان به عمل ملایم
۶. استفاده از کلمات مثبت و تشویق‌کننده
۷. نگارش روان و بدون غلط املایی
۸. مناسب برای توضیح محصول در سایت‌های فروشگاهی

نمونه‌های توضیح کوتاه:

محصول: "چای سبک بهاره لاهیجان"
توضیح: "چای سبک بهاره لاهیجان، با برگ‌های تازه و عطر بی‌نظیر، تجربه‌ای لذیذ از یک چای اصیل ایرانی را برای شما به ارمغان می‌آورد. این چای به روش سنتی تولید شده و حاوی آنتی‌اکسیدان‌های طبیعی است. برای پذیرایی از مهمانان یا لذت بردن از یک فنجان چای در زمان استراحت ایده‌آل است."

محصول: "موس ضد تعریق مردانه نیوآ"
توضیح: "موس ضد تعریق نیوآ با فرمول پیشرفته، تا ۴۸ ساعت از تعریق و بوی نامطبوع جلوگیری می‌کند. حاوی ویتامین E برای نرمی پوست و فاقد الکل برای جلوگیری از حساسیت. مناسب برای استفاده روزانه و در تمام فصول سال. اعتماد به نفس خود را با نیوآ تجربه کنید."

محصول: "کوله پشتی مدرسه طرح اسپایدرمن"
توضیح: "کوله پشتی مدرسه با طرح اسپایدرمن، همراه همیشگی کودکان برای روزهای مدرسه. جنس ضدآب، بندهای قابل تنظیم و چندین جیب برای سازماندهی وسایل. با طراحی شاد و رنگ‌های جذاب، باعث علاقه کودک به مدرسه رفتن می‌شود. هدیه‌ای مناسب برای دانش‌آموزان عزیز."

حالا برای محصول زیر یک توضیح کوتاه بنویس:"""

    user_prompt = f"""عنوان بهینه شده: {product_title}
عنوان اصلی: {raw_title}

لطفاً یک توضیح کوتاه جذاب بین ۳۵۰ تا ۴۵۰ کاراکتر بنویسید."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    return call_ollama(messages, temperature=0.7, max_tokens=200)

def generate_keywords(product_title: str, raw_title: str, short_description: str) -> Optional[str]:
    """مدل ۳: تولید کلمات کلیدی"""
    
    system_prompt = """شما یک متخصص سئو و بهینه‌سازی کلمات کلیدی هستید. وظیفه شما تولید کلمات کلیدی مرتبط برای محصولات است.

قوانین تولید کلمات کلیدی:
۱. تولید ۸-۱۲ کلمه کلیدی
۲. کلمات کلیدی باید شامل موارد زیر باشد:
   - کلمه کلیدی اصلی (طولانی)
   - کلمات کلیدی ثانویه
   - کلمات مرتبط با برند
   - کلمات توصیفی محصول
۳. کلمات باید به زبان فارسی و مرتبط با محصول باشند
۴. از کلمات عمومی و بی‌ربط پرهیز شود
۵. کلمات را با کاما (،) جدا کن
۶. کلمات تکراری نباشند

فرمت خروجی:
کلمه کلیدی اصلی، کلمه کلیدی ثانویه، کلمه مرتبط ۱، کلمه مرتبط ۲، ...

نمونه‌های خروجی:

برای "کفش ورزشی نایک ایرمکس":
کفش نایک ایرمکس، کفش ورزشی مردانه، نایک اصل، کفش رانینگ، کفش پیاده‌روی، کتانی نایک، کفش اسپرت

برای "قهوه ترک عربیکا":
قهوه ترک اصل، قهوه عربیکا، قهوه دم‌آوری، قهوه درجه یک، قهوه لذیذ، قهوه ایرانی، قهوه خانگی

برای "شامپو کلیر ضد ریزش":
شامپو ضد ریزش مو، شامپو کلیر، شامپو حجم دهنده، تقویت مو، شامپو پرپشت، مراقبت مو، شامپو ایرانی

برای "ماست کم چرب میهن":
ماست میهن، ماست کم چرب، ماست ۱.۵%، ماست پروبیوتیک، لبنیات میهن، ماست سبک، ماست خوشمزه

حالا برای محصول زیر کلمات کلیدی تولید کن:"""

    user_prompt = f"""عنوان محصول: {product_title}
عنوان اصلی: {raw_title}
توضیح کوتاه: {short_description}

کلمات کلیدی مناسب برای این محصول را تولید کن."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    return call_ollama(messages, temperature=0.5, max_tokens=150)

async def process_product_description(request_id: str, product_text: str):
    """پردازش غیرهمزمان برای تولید محتوای محصول"""
    global processing_results
    start_time = datetime.now()
    
    try:
        # مرحله ۱: استخراج عنوان
        logger.info(f" [{request_id}] مرحله ۱: استخراج عنوان...")
        processing_results[request_id]["progress"] = 0.2
        
        optimized_title = extract_product_title(product_text)
        if not optimized_title:
            raise Exception("خطا در استخراج عنوان")
        
        processing_results[request_id]["optimized_title"] = optimized_title
        processing_results[request_id]["progress"] = 0.4
        
        # مرحله ۲: تولید توضیح کوتاه
        logger.info(f" [{request_id}] مرحله ۲: تولید توضیح کوتاه...")
        processing_results[request_id]["progress"] = 0.5
        
        short_description = generate_short_description(optimized_title, product_text)
        if not short_description:
            raise Exception("خطا در تولید توضیح کوتاه")
        
        processing_results[request_id]["short_description"] = short_description
        processing_results[request_id]["progress"] = 0.7
        
        # مرحله ۳: تولید کلمات کلیدی
        logger.info(f" [{request_id}] مرحله ۳: تولید کلمات کلیدی...")
        processing_results[request_id]["progress"] = 0.8
        
        keywords = generate_keywords(optimized_title, product_text, short_description)
        if not keywords:
            raise Exception("خطا در تولید کلمات کلیدی")
        
        processing_results[request_id]["keywords"] = keywords
        processing_results[request_id]["progress"] = 1.0
        
        # محاسبه زمان پردازش
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        processing_results[request_id]["processing_time"] = processing_time
        processing_results[request_id]["status"] = "completed"
        
        logger.info(f" [{request_id}] پردازش کامل شد در {processing_time:.2f} ثانیه")
        
    except Exception as e:
        logger.error(f" [{request_id}] خطا در پردازش: {e}")
        processing_results[request_id]["status"] = "failed"
        processing_results[request_id]["error"] = str(e)

# API Endpoints
@app.get("/")
async def root():
    """صفحه اصلی API"""
    return {
        "message": "Product Description Generator API",
        "version": "1.0.0",
        "endpoints": {
            "POST /generate": "ایجاد محتوای جدید برای محصول",
            "GET /status/{request_id}": "بررسی وضعیت پردازش",
            "GET /result/{request_id}": "دریافت نتایج پردازش"
        },
        "docs": "/docs"
    }

@app.post("/generate", response_model=ProductResponse, status_code=202)
async def generate_description(product_request: ProductRequest, background_tasks: BackgroundTasks):
    """
    ایجاد درخواست جدید برای تولید محتوای محصول
    
    دریافت متن محصول و شروع پردازش غیرهمزمان
    """
    global processing_results
    # ایجاد شناسه درخواست
    request_id = product_request.request_id or str(uuid.uuid4())
    
    # ذخیره وضعیت اولیه
    processing_results[request_id] = {
        "status": "processing",
        "progress": 0.0,
        "created_at": datetime.now(),
        "product_text": product_request.text,
        "optimized_title": None,
        "short_description": None,
        "keywords": None,
        "processing_time": None
    }
    
    # شروع پردازش در پس‌زمینه
    background_tasks.add_task(process_product_description, request_id, product_request.text)
    
    logger.info(f"درخواست جدید ایجاد شد: {request_id}")
    
    return ProductResponse(
        request_id=request_id,
        status="processing",
        created_at=processing_results[request_id]["created_at"],
        processing_time=None
    )

@app.get("/status/{request_id}", response_model=StatusResponse)
async def get_status(request_id: str):
    """
    بررسی وضعیت پردازش یک درخواست
    
    نمایش پیشرفت پردازش
    """
    global processing_results
    print(processing_results)
    if request_id not in processing_results:
        raise HTTPException(status_code=404, detail="شناسه درخواست یافت نشد")
    
    result = processing_results[request_id]
    
    # محاسبه زمان تخمینی باقی‌مانده
    estimated_time = None
    if result["status"] == "processing":
        elapsed = (datetime.now() - result["created_at"]).total_seconds()
        if result["progress"] > 0:
            estimated_time = (elapsed / result["progress"]) * (1 - result["progress"])
    
    return StatusResponse(
        request_id=request_id,
        status=result["status"],
        progress=result["progress"],
        estimated_time=estimated_time
    )

@app.get("/result/{request_id}", response_model=ProductResponse)
async def get_result(request_id: str):
    """
    دریافت نتایج پردازش
    
    در صورت تکمیل پردازش، نتایج کامل را برمی‌گرداند
    """
    global processing_results
    if request_id not in processing_results:
        raise HTTPException(status_code=404, detail="شناسه درخواست یافت نشد")
    
    result = processing_results[request_id]
    
    if result["status"] == "processing":
        raise HTTPException(status_code=202, detail="پردازش هنوز کامل نشده است")
    
    if result["status"] == "failed":
        raise HTTPException(
            status_code=500,
            detail=f"خطا در پردازش: {result.get('error', 'خطای ناشناخته')}"
        )
    
    return ProductResponse(
        request_id=request_id,
        status=result["status"],
        optimized_title=result["optimized_title"],
        short_description=result["short_description"],
        keywords=result["keywords"],
        created_at=result["created_at"],
        processing_time=result["processing_time"]
    )

@app.get("/health")
async def health_check():
    """بررسی سلامت API و اتصال به Ollama"""
    try:
        # بررسی اتصال به Ollama
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=50)
        
        if response.status_code == 200:
            return {
                "status": "healthy",
                "ollama_connected": True,
                "model": MODEL_NAME,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "unhealthy",
                "ollama_connected": False,
                "error": f"Ollama returned {response.status_code}",
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        return {
            "status": "unhealthy",
            "ollama_connected": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/examples")
async def get_examples():
    """دریافت مثال‌های آماده برای تست"""
    examples = [
        {
            "text": "برنج صدری دمسیاه درجه یک اعلا محصول 1404 برنجکوبی قلی پور بسته بندی 10 کیلویی",
            "description": "برنج ایرانی مرغوب"
        },
        {
            "text": "چای سبک بهاره لاهیجان ممتاز محصول 1403 بسته بندی 500 گرمی",
            "description": "چای شمال درجه یک"
        },
        {
            "text": "روغن کنجد ارگانیک فشار سرد اصل بسته 900 میلی لیتر",
            "description": "روغن کنجد ارگانیک"
        },
        {
            "text": "کفش ورزشی نایک ایرمکس مشکی سایز 42 مدل 2024 اصل",
            "description": "کفش ورزشی نایک"
        },
        {
            "text": "شامپو حجم دهنده پرپشت کلیر ضد ریزش مو بسته 400 میلی لیتر",
            "description": "شامپو ضد ریزش مو"
        }
    ]
    
    return {"examples": examples}

@app.delete("/result/{request_id}")
async def delete_result(request_id: str):
    """
    حذف نتایج یک درخواست
    
    آزادسازی حافظه
    """
    global processing_results
    if request_id in processing_results:
        del processing_results[request_id]
        logger.info(f" نتایج حذف شد: {request_id}")
        return {"message": "نتایج با موفقیت حذف شدند", "request_id": request_id}
    else:
        raise HTTPException(status_code=404, detail="شناسه درخواست یافت نشد")

# Middleware برای لاگ‌گیری درخواست‌ها
@app.middleware("http")
async def log_requests(request, call_next):
    logger.info(f"{request.method} {request.url.path}")
    
    try:
        response = await call_next(request)
        logger.info(f"{request.method} {request.url.path} - {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"{request.method} {request.url.path} - {str(e)}")
        raise

# اجرای سرور (فقط در صورت اجرای مستقیم)
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        workers = 1  # استفاده از 1 worker با 16 thread داخلی
    )