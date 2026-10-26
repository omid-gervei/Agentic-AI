import streamlit as st
import requests
import json
import time

# Ollama configuration
OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "gemma3:1b"  # یا هر مدل دیگری که دارید

# Page configuration
st.set_page_config(
    page_title="تولید کننده محتوای محصول",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Persian/RTL support
st.markdown("""
<style>
html, body { direction: rtl; }
    .main .block-container {
        direction: rtl;
        text-align: right;
        font-family: 'Vazir', 'Tahoma', sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        text-align: right;
        font-family: 'Vazir', 'Tahoma', sans-serif;
    }
    .stTextInput > div > div > input {
        text-align: right;
        font-family: 'Vazir', 'Tahoma', sans-serif;
    }
    .stTextArea > div > div > textarea {
        text-align: right;
        font-family: 'Vazir', 'Tahoma', sans-serif;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 20px;
        margin: 10px 0;
        text-align: right;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 20px;
        margin: 10px 0;
        text-align: right;
    }
    .processing-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 20px;
        margin: 10px 0;
        text-align: right;
    }
</style>
""", unsafe_allow_html=True)

def call_ollama(messages, temperature=0.7, max_tokens=500):
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
            timeout=6000
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("message", {}).get("content", "").strip()
        else:
            st.error(f"خطا در ارتباط با مدل: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"خطا در ارتباط با Ollama: {e}")
        return None

def extract_product_title(raw_title):
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

ورودی: من برنج دم سیاه دارم. ۱۰۰ کیلیو هر کلیو ۸۰۰ تومن
خروجی: برنج دم سیاه اعلا

حالا عنوان زیر را بر اساس قوانین فوق بهینه کن:"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": raw_title}
    ]
    
    return call_ollama(messages, temperature=0.3, max_tokens=100)

def generate_short_description(product_title, raw_title):
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

def generate_keywords(product_title, raw_title, short_description):
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

def main():
    # Header
    st.title(" تولید کننده هوشمند محتوای محصول")
    st.markdown("### تبدیل عنوان محصول به محتوای کامل بازاریابی")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header(" تنظیمات")
        
        # Model selection
        model_options = ["gemma3:1b", "llama3.2:3b", "mistral", "qwen2:7b"]
        selected_model = st.selectbox(
            "انتخاب مدل:",
            model_options,
            index=0
        )
        
        # Update model name
        global MODEL_NAME
        MODEL_NAME = selected_model
        
        st.markdown("---")
        
        # Examples
        st.header(" مثال‌های آماده")
        
        example_titles = [
            "برنج صدری دمسیاه درجه یک اعلا محصول 1404 برنجکوبی قلی پور بسته بندی 10 کیلویی",
            "چای سبک بهاره لاهیجان ممتاز محصول 1403 بسته بندی 500 گرمی",
            "روغن کنجد ارگانیک فشار سرد اصل بسته 900 میلی لیتر",
            "کفش ورزشی نایک ایرمکس مشکی سایز 42 مدل 2024 اصل",
            "شامپو حجم دهنده پرپشت کلیر ضد ریزش مو بسته 400 میلی لیتر"
        ]
        
        for example in example_titles:
            if st.button(f" {example[:40]}...", key=f"example_{example[:10]}"):
                st.session_state.product_title_input = example
        
        st.markdown("---")
        
        # Instructions
        st.info("""
        ** راهنما:**
        1. عنوان محصول را وارد کنید
        2. دکمه تولید را بزنید
        3. منتظر بمانید تا هر سه مدل اجرا شوند
        4. از نتایج کپی کنید
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Product title input
        st.subheader(" وارد کردن عنوان محصول")
        
        product_title_input = st.text_area(
            "عنوان محصول خود را وارد کنید:",
            height=100,
            placeholder="مثال: برنج صدری دمسیاه درجه یک اعلا محصول 1404 برنجکوبی قلی پور بسته بندی 10 کیلویی",
            key="product_title_input"
        )
        
        # Generate button
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            generate_all = st.button(
                " تولید همه موارد",
                type="primary",
                use_container_width=True
            )
        
        with col_btn2:
            generate_title_only = st.button(
                " فقط عنوان",
                use_container_width=True
            )
        
        with col_btn3:
            if 'results' in st.session_state:
                clear_results = st.button(
                    " پاک کردن نتایج",
                    use_container_width=True
                )
                if clear_results:
                    st.session_state.pop('results', None)
                    st.rerun()
        
        st.markdown("---")
    
    with col2:
        st.subheader(" آمار تولید")
        
        if 'results' in st.session_state:
            results = st.session_state.results
            
            if 'optimized_title' in results:
                title_length = len(results['optimized_title'])
                st.metric("طول عنوان", f"{title_length} کاراکتر")
            
            if 'short_description' in results:
                desc_length = len(results['short_description'])
                st.metric("طول توضیح کوتاه", f"{desc_length} کاراکتر")
                
                # Check if within range
                if 350 <= desc_length <= 450:
                    st.success(" طول مناسب")
                else:
                    st.warning(" خارج از محدوده ۳۵۰-۴۵۰")
            
            if 'keywords' in results:
                keywords_count = len(results['keywords'].split('،'))
                st.metric("تعداد کلمات کلیدی", keywords_count)
        
        else:
            st.info("هنوز محتوایی تولید نشده است")
    
    # Processing and results section
    if generate_all or generate_title_only:
        if not product_title_input.strip():
            st.error(" لطفاً عنوان محصول را وارد کنید")
            return
        
        # Initialize session state for results
        if 'results' not in st.session_state:
            st.session_state.results = {}
        
        # Display processing
        processing_container = st.container()
        
        with processing_container:
            if generate_all:
                st.subheader(" در حال پردازش...")
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Extract title
                status_text.text("مرحله ۱/۳: استخراج عنوان مناسب...")
                progress_bar.progress(0.2)
                
                optimized_title = extract_product_title(product_title_input)
                
                if optimized_title:
                    st.session_state.results['optimized_title'] = optimized_title
                    progress_bar.progress(0.4)
                    time.sleep(0.5)
                    
                    # Step 2: Generate short description
                    status_text.text("مرحله ۲/۳: تولید توضیح کوتاه...")
                    progress_bar.progress(0.6)
                    
                    short_description = generate_short_description(optimized_title, product_title_input)
                    
                    if short_description:
                        st.session_state.results['short_description'] = short_description
                        progress_bar.progress(0.8)
                        time.sleep(0.5)
                        
                        # Step 3: Generate keywords
                        status_text.text("مرحله ۳/۳: تولید کلمات کلیدی...")
                        progress_bar.progress(0.9)
                        
                        keywords = generate_keywords(optimized_title, product_title_input, short_description)
                        
                        if keywords:
                            st.session_state.results['keywords'] = keywords
                            progress_bar.progress(1.0)
                            status_text.text(" پردازش کامل شد!")
                            time.sleep(1)
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
            elif generate_title_only:
                st.subheader(" در حال پردازش...")
                
                with st.spinner("در حال استخراج عنوان مناسب..."):
                    optimized_title = extract_product_title(product_title_input)
                    
                    if optimized_title:
                        st.session_state.results['optimized_title'] = optimized_title
    
    # Display results if available
    if 'results' in st.session_state and st.session_state.results:
        results = st.session_state.results
        
        st.markdown("---")
        st.subheader(" نتایج تولید شده")
        
        # Display in tabs
        tab1, tab2, tab3 = st.tabs([" عنوان بهینه", " توضیح کوتاه", " کلمات کلیدی"])
        
        with tab1:
            if 'optimized_title' in results:
                st.markdown("### عنوان بهینه شده")
                st.markdown(f'<div class="success-box"><h3>{results["optimized_title"]}</h3></div>', unsafe_allow_html=True)
                
                # Copy button for title
                col_copy1, col_stats1 = st.columns([1, 2])
                with col_copy1:
                    if st.button(" کپی عنوان", key="copy_title"):
                        st.success("عنوان کپی شد!")
                
                with col_stats1:
                    st.caption(f"طول: {len(results['optimized_title'])} کاراکتر")
        
        with tab2:
            if 'short_description' in results:
                st.markdown("### توضیح کوتاه محصول")
                st.markdown(f'<div class="info-box">{results["short_description"]}</div>', unsafe_allow_html=True)
                
                # Character count
                desc_length = len(results['short_description'])
                
                col_copy2, col_stats2 = st.columns([1, 2])
                with col_copy2:
                    if st.button(" کپی توضیح", key="copy_desc"):
                        st.success("توضیح کپی شد!")
                
                with col_stats2:
                    if 350 <= desc_length <= 450:
                        st.success(f" طول مناسب: {desc_length} کاراکتر")
                    else:
                        st.warning(f" طول: {desc_length} کاراکتر (محدوده ایده‌آل: ۳۵۰-۴۵۰)")
        
        with tab3:
            if 'keywords' in results:
                st.markdown("### کلمات کلیدی")
                st.markdown(f'<div class="processing-box">{results["keywords"]}</div>', unsafe_allow_html=True)
                
                # Split and display as badges
                keywords_list = [k.strip() for k in results["keywords"].split('،') if k.strip()]
                
                st.markdown("#### نمایش به صورت برچسب:")
                cols = st.columns(4)
                for idx, keyword in enumerate(keywords_list):
                    with cols[idx % 4]:
                        st.markdown(f'<span style="background-color: #4CAF50; color: white; padding: 5px 10px; border-radius: 15px; margin: 2px; display: inline-block;">{keyword}</span>', unsafe_allow_html=True)
                
                col_copy3, col_stats3 = st.columns([1, 2])
                with col_copy3:
                    if st.button(" کپی کلمات کلیدی", key="copy_keywords"):
                        st.success("کلمات کلیدی کپی شد!")
                
                with col_stats3:
                    st.caption(f"تعداد: {len(keywords_list)} کلمه کلیدی")
        
        # Export options
        st.markdown("---")
        st.subheader(" خروجی نهایی")
        
        col_export1, col_export2, col_export3 = st.columns(3)
        
        with col_export1:
            if st.button(" ذخیره در فایل", use_container_width=True):
                # Create a text file with all results
                export_content = f"""نتایج تولید محتوای محصول
تاریخ تولید: {time.strftime('%Y-%m-%d %H:%M:%S')}
مدل استفاده شده: {MODEL_NAME}
---
عنوان ورودی: {product_title_input}
---
 عنوان بهینه شده:
{results.get('optimized_title', '')}
---
 توضیح کوتاه محصول:
{results.get('short_description', '')}
---
 کلمات کلیدی:
{results.get('keywords', '')}
"""
                
                filename = f"product_content_{int(time.time())}.txt"
                st.download_button(
                    label="دانلود فایل متن",
                    data=export_content,
                    file_name=filename,
                    mime="text/plain"
                )
        
        with col_export2:
            # JSON export
            export_json = json.dumps({
                "input_title": product_title_input,
                "optimized_title": results.get('optimized_title', ''),
                "short_description": results.get('short_description', ''),
                "keywords": results.get('keywords', ''),
                "generated_at": time.strftime('%Y-%m-%d %H:%M:%S'),
                "model": MODEL_NAME
            }, ensure_ascii=False, indent=2)
            
            st.download_button(
                label=" دانلود JSON",
                data=export_json,
                file_name=f"product_content_{int(time.time())}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col_export3:
            # HTML export
            if all(k in results for k in ['optimized_title', 'short_description', 'keywords']):
                html_content = f"""<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>محتوای تولید شده برای محصول</title>
    <style>
        body {{
            direction: rtl;
            font-family: 'Tahoma', 'Arial', sans-serif;
            line-height: 1.8;
            margin: 0;
            padding: 20px;
            background-color: #f5f9fc;
            color: #333;
            text-align: right;
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        .section {{
            margin: 30px 0;
            padding: 20px;
            border-radius: 8px;
        }}
        .title-section {{ background-color: #d4edda; }}
        .desc-section {{ background-color: #d1ecf1; }}
        .keywords-section {{ background-color: #fff3cd; }}
        .keyword-badge {{
            background-color: #4CAF50;
            color: white;
            padding: 5px 10px;
            border-radius: 15px;
            margin: 5px;
            display: inline-block;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>محتوای تولید شده برای محصول</h1>
        <p><strong>تاریخ تولید:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>مدل:</strong> {MODEL_NAME}</p>
        
        <hr>
        
        <div class="section title-section">
            <h2> عنوان بهینه شده</h2>
            <h3>{results['optimized_title']}</h3>
        </div>
        
        <div class="section desc-section">
            <h2> توضیح کوتاه محصول</h2>
            <p>{results['short_description']}</p>
            <p><em>طول: {len(results['short_description'])} کاراکتر</em></p>
        </div>
        
        <div class="section keywords-section">
            <h2> کلمات کلیدی</h2>
            <p>{results['keywords']}</p>
            <div>
                {" ".join([f'<span class="keyword-badge">{k}</span>' for k in results['keywords'].split('،')])}
            </div>
        </div>
    </div>
</body>
</html>"""
                
                st.download_button(
                    label=" دانلود HTML",
                    data=html_content,
                    file_name=f"product_content_{int(time.time())}.html",
                    mime="text/html",
                    use_container_width=True
                )
    

if __name__ == "__main__":
    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = {}
    
    main()