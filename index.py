from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS
import numpy as np
import os
import uuid
import re

# ================== APP ==================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== MODELS ==================
class Question(BaseModel):
    question: str

class GuestRequest(BaseModel):
    guest_index: int

# ================== AUDIO ==================
AUDIO_DIR = "audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

# Maximum number of audio files to keep
MAX_AUDIO_FILES = 50

def cleanup_old_audio_files():
    """Delete oldest audio files when limit is exceeded"""
    try:
        audio_files = [
            os.path.join(AUDIO_DIR, f) 
            for f in os.listdir(AUDIO_DIR) 
            if f.endswith('.mp3')
        ]
        
        if len(audio_files) > MAX_AUDIO_FILES:
            # Sort by creation time
            audio_files.sort(key=lambda x: os.path.getctime(x))
            
            # Delete oldest files
            files_to_delete = len(audio_files) - MAX_AUDIO_FILES
            for file_path in audio_files[:files_to_delete]:
                os.remove(file_path)
                print(f"🗑️ Cleaned up old audio: {os.path.basename(file_path)}")
    except Exception as e:
        print(f"Error during cleanup: {e}")

# ================== GUEST DATA ==================
GUEST_INFO = [
    {
        "name": "Shri Danish Ashraf",
        "role": "Inaugural Ceremony – Chief Guest",
        "text": "We are honoured to welcome Shri Danish Ashraf, Joint Development Commissioner, Ministry of MSME, Government of India, as the Chief Guest for the Inaugural Ceremony. His distinguished service and leadership in strengthening India's MSME ecosystem have significantly contributed to innovation, entrepreneurship, and inclusive economic growth. His presence adds great value and inspiration to the inauguration of the event."
    },
    {
        "name": "Lt. Gen. Anil Malik, AVSM",
        "role": "Inaugural Ceremony – Guest of Honour",
        "text": "We are privileged to have Lt. Gen. Anil Malik, AVSM, Former Director General, Discipline, Ceremonials & Welfare, Army Headquarters, as the Guest of Honour. With a remarkable career in the Indian Army, he exemplifies discipline, integrity, and selfless service. His vast experience and commitment to national service are a source of motivation for all."
    },
    {
        "name": "Dr. Sunil Kumar Barnwal",
        "role": "Valedictory Ceremony – Chief Guest",
        "text": "It is our distinct pleasure to welcome Dr. Sunil Kumar Barnwal, CEO, National Health Authority, Ministry of Health & Family Welfare, Government of India, as the Chief Guest for the Valedictory Ceremony. His visionary leadership in advancing India's digital health initiatives has played a crucial role in transforming the healthcare ecosystem of the nation. His insights will greatly enrich the concluding session of the event."
    },
    {
        "name": "Ms. Hena Parveen",
        "role": "Valedictory Ceremony – Guest of Honour",
        "text": "We are delighted to have Ms. Hena Parveen, Educationist & Advisor, UBA-JH, as the Guest of Honour. Her dedication to education, social upliftment, and community development reflects her deep commitment to nation-building. Her guidance and continued support have been instrumental in inspiring young minds and strengthening academic-social initiatives."
    }
]

# ================== FAQ MAPS ==================

GENERAL_FAQ = {
    "what is nerdz": {
        "triggers": [
            "what is nerdz",
            "what is nerdz 26",
            "what is nerdz fest",
            "nerdz fest",
            "tell me about nerdz",
            "about nerdz",
            "define nerdz",
            "nerdz meaning",
            "nerdz festival"
        ],
        "answer": "NERDZ '26 is the flagship techno-cultural fest of the School of Engineering Sciences & Technology (SEST) at Jamia Hamdard University."
    },

    "about nerdz": {
        "triggers": [
            "about nerdz",
            "about nerdz fest",
            "nerdz details",
            "nerdz information",
            "tell me about nerdz 26"
        ],
        "answer": "NERDZ '26 is a two-day techno-cultural fest combining technical competitions, cultural performances, music, theatre, and gaming."
    },

    "why nerdz": {
        "triggers": [
            "why nerdz",
            "why is it called nerdz",
            "meaning of nerdz",
            "why the name nerdz",
            "nerdz name meaning"
        ],
        "answer": "The name NERDZ redefines the meaning of a nerd—curious, creative, expressive, and unapologetically fun."
    },

    "organizers": {
        "triggers": [
            "who organizes nerdz",
            "who organised nerdz",
            "organizer of nerdz",
            "nerdz organizer",
            "who conducts nerdz"
        ],
        "answer": "NERDZ is organized by the School of Engineering Sciences & Technology (SEST), Jamia Hamdard University."
    },

    "participation": {
        "triggers": [
            "who can participate",
            "who can join nerdz",
            "can outsiders participate",
            "eligibility for nerdz",
            "is nerdz open for all"
        ],
        "answer": "Students from Jamia Hamdard and other institutions can participate depending on event guidelines."
    }
}


DATE_FAQ = {
    "dates": {
        "triggers": [
            "when is nerdz",
            "nerdz date",
            "nerdz dates",
            "date of nerdz",
            "nerdz 2026 date",
            "when will nerdz happen"
        ],
        "answer": "NERDZ '26 takes place on February 5–6, 2026."
    },

    "duration": {
        "triggers": [
            "how many days",
            "duration of nerdz",
            "how long is nerdz",
            "number of days nerdz"
        ],
        "answer": "NERDZ '26 is a two-day fest."
    },

    "day1": {
        "triggers": [
            "day 1",
            "first day",
            "day one",
            "nerdz day 1 date"
        ],
        "answer": "Day 1 of NERDZ '26 is on February 5, 2026."
    },

    "day2": {
        "triggers": [
            "day 2",
            "second day",
            "day two",
            "nerdz day 2 date"
        ],
        "answer": "Day 2 of NERDZ '26 is on February 6, 2026."
    }
}


VENUE_FAQ = {
    "overall": {
        "triggers": [
            "where is nerdz",
            "venue",
            "nerdz venue",
            "location of nerdz",
            "where is nerdz held"
        ],
        "answer": "NERDZ '26 is held at Jamia Hamdard University campus. Events take place at the Convention Centre, Stage Area, Labs, Archives, and Ground."
    },

    "hackathon": {
        "triggers": [
            "hackathon venue",
            "where is hackathon",
            "hackathon location"
        ],
        "answer": "The Hackathon is held at Lab 401."
    },

    "star night": {
        "triggers": [
            "star night venue",
            "where is star night",
            "star night location"
        ],
        "answer": "Star Night is held at the Stage."
    },

    "bug fest": {
        "triggers": [
            "bug fest venue",
            "bug fest location",
            "where is bug fest"
        ],
        "answer": "Bug Fest takes place at the 1st Floor Lab."
    },

    "qawwali": {
        "triggers": [
            "qawwali venue",
            "qawwali night venue",
            "where is qawwali"
        ],
        "answer": "Qawwali Night is held at the Stage."
    }
}


CATEGORY_FAQ = {
    "technical": {
        "triggers": [
            "technical events",
            "coding events",
            "tech events",
            "programming events"
        ],
        "answer": "Technical events include Hackathon, Bug Fest, UI Design Competition, Tech Quiz, and 5G Ideathon."
    },

    "cultural": {
        "triggers": [
            "cultural events",
            "cultural programs",
            "non technical events"
        ],
        "answer": "Cultural events include Dance Performance, Singing, Theatre, Bait Bazi, and Nukkad Natak."
    },

    "music": {
        "triggers": [
            "music events",
            "singing events",
            "concerts"
        ],
        "answer": "Music events include Star Night, Battle of Bands, and Qawwali Night."
    },

    "gaming": {
        "triggers": [
            "gaming events",
            "games",
            "esports"
        ],
        "answer": "Gaming events include the Battle Zone Gaming Competition."
    }
}

GUEST_FAQ = {
    "guests": {
        "triggers": [
            "who are the guests",
            "chief guest",
            "guest of honour",
            "distinguished guests",
            "inaugural guest",
            "valedictory guest",
            "tell me about guests",
            "guest speakers",
            "special guests"
        ],
        "answer": """Our distinguished guests for NERDZ '26:

INAUGURAL CEREMONY:
• Chief Guest: Shri Danish Ashraf - Joint Development Commissioner, Ministry of MSME, Government of India
• Guest of Honour: Lt. Gen. Anil Malik, AVSM - Former Director General, Indian Army

VALEDICTORY CEREMONY:
• Chief Guest: Dr. Sunil Kumar Barnwal - CEO, National Health Authority
• Guest of Honour: Ms. Hena Parveen - Educationist & Advisor, UBA-JH

Each brings exceptional expertise and inspiration to our event."""
    }
}


EVENT_TITLES = [
    "Hackathon",
    "5G Ideathon",
    "Treasure Hunt",
    "Solo and Duet Singing Competition",
    "Dance Performance",
    "Bait Bazi",
    "Nukkad Natak",
    "Star Night",
    "UI Design Competition",
    "Bug Fest",
    "Tech Quiz",
    "Battle Zone Gaming Competition",
    "Special Theatre Performance by SEST",
    "Battle of Bands",
    "Valedictory Ceremony",
    "Qawwali Night"
]

# ================== UTILS ==================

def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower())

def match_faq(question, faq_dict):
    for item in faq_dict.values():
        for trigger in item["triggers"]:
            if trigger in question:
                return item["answer"]
    return None


def extract_event_block(event_name: str, text: str):
    pattern = rf"{event_name}(.+?)(?=\n[A-Z]|$)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(0).strip() if match else None

def smart_chunk_text(text):
    chunks = []
    events = re.split(r'\n(?=[A-Z].*\n)', text)

    for event in events:
        event = event.strip()
        if len(event) > 30:
            chunks.append(event)

    return list(dict.fromkeys(chunks))

# ================== TF-IDF ==================

SIMILARITY_THRESHOLD = 0.15

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=5000
)

# ================== GUEST AUDIO ENDPOINT ==================

@app.post("/guest-audio")
async def get_guest_audio(data: GuestRequest):
    """Generate audio for carousel guest information"""
    print(f"Guest audio request for index: {data.guest_index}")
    
    if data.guest_index < 0 or data.guest_index >= len(GUEST_INFO):
        return {"text": "Invalid guest index", "audio": None}
    
    guest = GUEST_INFO[data.guest_index]
    full_text = f"{guest['role']}. {guest['text']}"
    
    return create_response(full_text)

# ================== MAIN ENDPOINT ==================

@app.post("/ask")
async def ask_bot(data: Question):
    print(f"Received question: {data.question}")
    
    if not data.question.strip():
        return {"text": "Please ask a valid question 🙂", "audio": None}

    question = normalize(data.question)
    print(f"Normalized question: {question}")

    try:
        with open("nerdz_info.txt", "r", encoding="utf-8") as f:
            document = f.read()
    except FileNotFoundError:
        print("ERROR: nerdz_info.txt not found")
        return {"text": "Knowledge file not found. Please contact administrator.", "audio": None}

    # 1️⃣ GENERAL FAQ
    answer = match_faq(question, GENERAL_FAQ)
    if answer:
        print(f"Matched GENERAL_FAQ: {answer[:50]}...")
        return create_response(answer)

    # 2️⃣ DATE FAQ
    answer = match_faq(question, DATE_FAQ)
    if answer:
        print(f"Matched DATE_FAQ: {answer[:50]}...")
        return create_response(answer)

    # 3️⃣ VENUE FAQ
    answer = match_faq(question, VENUE_FAQ)
    if answer:
        print(f"Matched VENUE_FAQ: {answer[:50]}...")
        return create_response(answer)

    # 4️⃣ CATEGORY FAQ
    answer = match_faq(question, CATEGORY_FAQ)
    if answer:
        print(f"Matched CATEGORY_FAQ: {answer[:50]}...")
        return create_response(answer)

    # 5️⃣ GUEST FAQ
    answer = match_faq(question, GUEST_FAQ)
    if answer:
        print(f"Matched GUEST_FAQ")
        return create_response(answer)

    # 6️⃣ EVENT EXTRACTION
    for event in EVENT_TITLES:
        if event.lower() in question:
            block = extract_event_block(event, document)
            if block:
                print(f"Matched EVENT: {event}")
                return create_response(block)

    # 7️⃣ TF-IDF FALLBACK
    chunks = smart_chunk_text(document)
    tfidf = vectorizer.fit_transform(chunks + [question])
    sims = cosine_similarity(tfidf[-1], tfidf[:-1])[0]
    best_idx = np.argmax(sims)

    print(f"TF-IDF best similarity: {sims[best_idx]:.3f}")

    if sims[best_idx] < SIMILARITY_THRESHOLD:
        return create_response(
            "Sorry, I couldn't find relevant information for that question. Try asking about events, dates, venues, or our distinguished guests!"
        )

    return create_response(chunks[best_idx])

# ================== RESPONSE BUILDER ==================

def create_response(text: str, save_audio: bool = True):
    """
    Create response with optional audio saving
    save_audio=True: Save to disk (for user queries)
    save_audio=False: Don't save (for welcome messages, errors, etc.)
    """
    try:
        if not save_audio:
            # Don't generate audio for simple messages
            return {
                "text": text,
                "audio": None
            }
        
        # Cleanup old files before creating new one
        cleanup_old_audio_files()
        
        filename = f"audio_{uuid.uuid4().hex}.mp3"
        filepath = os.path.join(AUDIO_DIR, filename)

        print(f"Generating audio: {filename}")
        tts = gTTS(text=text, lang="en")
        tts.save(filepath)
        print(f"Audio saved successfully: {filepath}")

        response = {
            "text": text,
            "audio": filename
        }
        print(f"Returning response with audio: {filename}")
        return response
    except Exception as e:
        print(f"ERROR in create_response: {str(e)}")
        return {
            "text": text,
            "audio": None
        }

# ================== AUDIO SERVE ==================

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    path = os.path.join(AUDIO_DIR, filename)
    print(f"Audio request for: {filename}")
    
    if not os.path.exists(path):
        print(f"ERROR: Audio file not found: {path}")
        return {"error": "Audio not found"}
    
    print(f"Serving audio file: {path}")
    return FileResponse(
        path, 
        media_type="audio/mpeg",
        headers={
            "Cache-Control": "no-cache",
            "Access-Control-Allow-Origin": "*"
        }
    )

# ================== HEALTH ==================

@app.get("/")
async def root():
    return {
        "status": "NERDZ '26 Bot running 🚀",
        "mode": "Deterministic + TF-IDF fallback + Guest Carousel",
        "features": ["Chat Bot", "Guest Carousel", "Auto-play Audio"],
        "audio_files": len([f for f in os.listdir(AUDIO_DIR) if f.endswith('.mp3')])
    }

@app.post("/cleanup-audio")
async def manual_cleanup():
    """Manually trigger audio cleanup"""
    try:
        audio_files = [
            os.path.join(AUDIO_DIR, f) 
            for f in os.listdir(AUDIO_DIR) 
            if f.endswith('.mp3')
        ]
        
        count_before = len(audio_files)
        cleanup_old_audio_files()
        
        audio_files_after = [
            f for f in os.listdir(AUDIO_DIR) 
            if f.endswith('.mp3')
        ]
        count_after = len(audio_files_after)
        
        return {
            "status": "success",
            "deleted": count_before - count_after,
            "remaining": count_after
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.delete("/audio/all")
async def delete_all_audio():
    """Delete ALL audio files (use with caution)"""
    try:
        audio_files = [
            os.path.join(AUDIO_DIR, f) 
            for f in os.listdir(AUDIO_DIR) 
            if f.endswith('.mp3')
        ]
        
        for file_path in audio_files:
            os.remove(file_path)
        
        return {
            "status": "success",
            "deleted": len(audio_files),
            "message": "All audio files deleted"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}