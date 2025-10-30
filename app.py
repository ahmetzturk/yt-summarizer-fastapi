# ======================================================
# YouTube Summarizer (FastAPI + Gemini 2.5)
# ------------------------------------------------------
# 1️⃣ YouTube URL veya video ID alır
# 2️⃣ youtube-transcript-api ile transcript metnini çeker
# 3️⃣ google-genai (Gemini 2.5) ile metni özetler
# ======================================================

import os
import re
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


from yt_dlp import YoutubeDL

from dotenv import load_dotenv
from google import genai


# ======================================================
# CONFIG & SABİTLER
# ======================================================

# .env dosyasındaki ortam değişken adı
ENV_API_KEY_NAME = "GOOGLE_API_KEY"

# Varsayılan Gemini modeli (yeni SDK için)
DEFAULT_MODEL = "gemini-2.5-flash"

# Transcript dili önceliği (önce Türkçe, sonra İngilizce)
DEFAULT_LANGS = ["tr", "tr-TR", "en"]

# Uzun metinlerde karakter sınırı (yaklaşık 4–5k token civarı)
MAX_INPUT_CHARS = 12_000

# YouTube URL’lerinden video ID çekmek için regex desenleri
_YT_ID_PATTERNS = [
    r"v=([A-Za-z0-9_\-]{6,})",
    r"youtu\.be/([A-Za-z0-9_\-]{6,})",
    r"shorts/([A-Za-z0-9_\-]{6,})",
]


# ======================================================
# GEMINI VE FASTAPI BAŞLATMA
# ======================================================

# .env dosyasını yükle
load_dotenv()

# Gemini API anahtarını al
API_KEY = os.getenv(ENV_API_KEY_NAME)
if not API_KEY:
    raise RuntimeError(f"{ENV_API_KEY_NAME} is not set in .env")

# google-genai Client nesnesini oluştur (tek sefer)
client = genai.Client(api_key=API_KEY)

# YouTube transcript API tek instance (daha performanslı)
ytt_api = YouTubeTranscriptApi()

# FastAPI uygulaması
app = FastAPI(title="YouTube Summarizer (Gemini 2.5)", version="1.0.0")

# Geliştirme aşamasında CORS’u serbest bırak
# Üretimde (deploy ederken) allow_origins listesine sadece kendi domainini ekle
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ======================================================
# ŞEMA TANIMLARI
# ======================================================

class SummarizeReq(BaseModel):
    """
    API isteği (POST /summarize)
    Kullanıcıdan gelen JSON gövdesi bu yapıya uymalıdır.
    """
    url_or_id: str                     # YouTube URL veya doğrudan video ID
    language: Optional[str] = "tr"     # Özet dili (varsayılan Türkçe)


class SummarizeRes(BaseModel):
    """
    API yanıtı (JSON)
    Özetin yanı sıra video kimliği ve dil bilgisi döner.
    """
    video_id: str
    language: str
    summary: str


# ======================================================
# YARDIMCI FONKSİYONLAR
# ======================================================

def extract_video_id(url_or_id: str) -> str:
    """
    YouTube URL/ID içinden video ID çıkarır.
    Eğer kullanıcı doğrudan ID girdiyse (ör. "dQw4w9WgXcQ"), onu döndürür.
    """
    for p in _YT_ID_PATTERNS:
        m = re.search(p, url_or_id)
        if m:
            return m.group(1)

    # Eğer sadece ID verildiyse ve 6+ karakterlik alfasayısal bir yapıya sahipse kabul edilir
    if re.fullmatch(r"[A-Za-z0-9_\-]{6,}", url_or_id):
        return url_or_id

    raise ValueError("Video ID bulunamadı. Geçerli bir YouTube URL'si veya video ID gönderin.")


# -------- AŞAMA 1: URL -> Transkript (yt-dlp ile) --------
def fetch_transcript(url: str, prefer_langs: Optional[List[str]] = None) -> str:
    """
    yt-dlp kullanarak verilen YouTube URL’sinden transkript metnini döndürür.
    """

    # prefer_langs şu anda yt-dlp tarafından otomatik yönetilir
    # yt-dlp'nin sadece transkripti çekmesini sağlayan ayarlar
    ydl_opts = {
        'writesubtitles': True,             # Altyazıları indirmeyi etkinleştir
        'skip_download': True,              # Videoyu indirmeyi atla
        'subtitleslangs': prefer_langs or ['tr', 'en'], # Tercih edilen diller
        'subtitlesformat': 'srv3',          # Altyazı formatı
        'quiet': True,                      # Çıktıları gizle
        'noprogress': True,                 # İlerleme çubuğunu gizle
        'force_generic_extractor': True     # Genel çıkarıcıyı zorla
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            # Transkript verisini çekme (yt-dlp'nin biraz karmaşık bir yapısı var)
            subtitles = info.get('subtitles')
            if not subtitles:
                raise ValueError("Bu video için transkript bulunamadı veya kapalıdır.")

            # Tercih edilen dillerden birini seçme
            selected_sub = None
            for lang in prefer_langs or ['tr', 'en']:
                if subtitles.get(lang):
                    selected_sub = subtitles[lang][-1] # Genellikle son (en kaliteli) formatı seç
                    break

            if not selected_sub:
                raise ValueError("Tercih edilen dillerde (tr/en) transkript bulunamadı.")

            # Transkripti gerçekten çekme (bu, ayrı bir http isteğidir)
            sub_url = selected_sub['url']
            import requests
            transcript_data = requests.get(sub_url).text

            # Sadece metin kısmını temizleme (basit bir temizleme)
            # Bu kısım karmaşık olabileceği için, şimdilik basit bir temizleme yapalım:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(transcript_data)
            full_text = " ".join([elem.text for elem in root.findall('.//text') if elem.text])

            if not full_text.strip():
                 raise ValueError("Transkript içeriği boş.")

            return full_text

    except Exception as e:
        # yt-dlp hatalarını genel ValueError olarak yakalayalım
        raise ValueError(f"Transcript alınırken hata: {e}")


# NOT: extract_video_id fonksiyonunu silebilirsiniz, çünkü yt-dlp doğrudan URL ile çalışır. 
# Ancak kalsın da, URL'nin geçerliliğini kontrol etmeye yardımcı olur.


def build_prompt(text: str, language: str) -> str:
    """
    Gemini'ye gönderilecek prompt'u oluşturur.
    Burada metin kısaltma ve yönlendirme yapılır.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("Özetlenecek metin boş.")

    # Uzun metinlerde gereksiz kısmı kes
    if len(text) > MAX_INPUT_CHARS:
        text = text[:MAX_INPUT_CHARS] + " ..."

    # Model yönlendirmesi (basit ama yeterli)
    return (
        f"Summarize the following YouTube transcript in {language}. "
        f"Provide a detailed but clear summary of the content:\n\"\"\"{text}\"\"\""
    )


def summarize_with_gemini(text: str, language: str = "tr", model: str = DEFAULT_MODEL) -> str:
    """
    google-genai (yeni SDK) kullanarak özet oluşturur.
    """
    prompt = build_prompt(text, language)

    try:
        # Modelden özet isteği
        resp = client.models.generate_content(model=model, contents=prompt)

        # Yanıt boş gelirse hata fırlat
        summary = (getattr(resp, "text", None) or "").strip()
        if not summary:
            raise RuntimeError("Gemini boş yanıt döndürdü.")

        return summary

    except Exception as e:
        raise RuntimeError(f"Gemini çağrısı başarısız: {e}")


# ======================================================
# API ROUTES
# ======================================================

@app.get("/health")
def health():
    """Basit sağlık kontrolü (canlı mı?)"""
    return {"status": "ok"}


@app.post("/summarize", response_model=SummarizeRes)
def summarize(req: SummarizeReq):
    """
    Ana endpoint:
    1️⃣ YouTube URL veya video ID alır
    2️⃣ Transcript'i çeker
    3️⃣ Gemini 2.5 ile özet oluşturur
    4️⃣ JSON döner
    """
    try:
        # --- Aşama 1: Transcript çek ---
        transcript = fetch_transcript(
            req.url_or_id,
            prefer_langs=[req.language or "tr", "tr-TR", "en"],
            preserve_formatting=False,
        )

        # --- Aşama 2: Özet oluştur ---
        vid = extract_video_id(req.url_or_id)
        summary = summarize_with_gemini(transcript, language=req.language or "tr")

        # --- Başarılı yanıt ---
        return SummarizeRes(video_id=vid, language=req.language or "tr", summary=summary)

    # Bilinen YouTube hataları (404)
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
        raise HTTPException(status_code=404, detail=f"Transcript bulunamadı veya devre dışı: {e}")

    # Geçersiz veri (örneğin bozuk URL)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Dış servis (Gemini) veya özetleme hatası
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    # Diğer beklenmeyen hatalar
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Beklenmeyen hata: {e}")
