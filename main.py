import os
import shutil
import uuid
import asyncio
import time
import re
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from symspellpy import SymSpell, Verbosity
import noisereduce as nr
import soundfile as sf

# 0. CPU Thread Optimization
torch.set_num_threads(4)

app = FastAPI(title="SAKLOLO / TOLONG Emergency API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "saved_audio")
os.makedirs(AUDIO_DIR, exist_ok=True)
app.mount("/static/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

# 1. Pre-load faster-whisper small model
print("Loading faster-whisper model (small, int8 CPU)...")
stt_model = WhisperModel("small", device="cpu", compute_type="int8")

# 2. Pre-load saklolo_bert_model
BERT_MODEL_PATH = os.path.join(BASE_DIR, "saklolo_bert_model")
print(f"Loading BERT model from {BERT_MODEL_PATH}...")

try:
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH, use_fast=True)
    bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
    bert_model.eval()
    print("Custom saklolo_bert_model loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load saklolo_bert_model. Error: {e}")
    tokenizer = None
    bert_model = None

# 3. Pre-load SymSpell Tagalog Lexicon
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
DICT_PATH = os.path.join(BASE_DIR, "tagalog_dict.txt")

if os.path.exists(DICT_PATH):
    sym_spell.load_dictionary(DICT_PATH, term_index=0, count_index=1, separator=" ")
    print("Tagalog Lexicon loaded into SymSpell memory!")
else:
    print("Warning: tagalog_dict.txt not found. SymSpell layer disabled.")

CATEGORY_MAP = {
    0: "MEDIKAL / AMBULANCE",
    1: "SUNOG / FIRE",
    2: "PULIS / POLICE",
    3: "DISASTER / LAHAR / FLOOD",
    "LABEL_0": "MEDIKAL / AMBULANCE",
    "LABEL_1": "SUNOG / FIRE",
    "LABEL_2": "PULIS / POLICE",
    "LABEL_3": "DISASTER / LAHAR / FLOOD"
}

TAGALOG_CONTEXT_FIXES = {
    r"\bsa kaya\b": "sa kalye",
    r"\bnaaksidenting\b": "naaksidenteng",
    r"\bbunggun\b": "nabangga",
    r"\bbunggon\b": "nabangga",
    r"\bkoche\b": "kotse",
    r"\bkotche\b": "kotse",
    r"\bmaypulis\b": "may pulis",
    r"\bambulansia\b": "ambulansya",
}

def denoise_audio_file(input_file_path: str) -> str:
    try:
        data, rate = sf.read(input_file_path)
        cleaned_data = nr.reduce_noise(y=data, sr=rate, prop_decrease=0.8)
        cleaned_path = input_file_path.replace(".aac", "_clean.wav")
        sf.write(cleaned_path, cleaned_data, rate)
        return cleaned_path
    except Exception as e:
        print(f"[NOISE CANCEL] Skipped noise reduction: {e}")
        return input_file_path

def normalize_context_phrases(text: str) -> str:
    for typo_pattern, correction in TAGALOG_CONTEXT_FIXES.items():
        text = re.sub(typo_pattern, correction, text, flags=re.IGNORECASE)
    return text

def spellcheck_tagalog_words(text: str) -> str:
    if not sym_spell.word_count:
        return text

    words = text.split()
    corrected_words = []
    
    for word in words:
        clean_word = re.sub(r'[^\w\s]', '', word).lower()
        if not clean_word or len(clean_word) <= 2:
            corrected_words.append(word)
            continue

        suggestions = sym_spell.lookup(clean_word, Verbosity.TOP, max_edit_distance=2)
        if suggestions and suggestions[0].distance > 0:
            corrected_words.append(suggestions[0].term)
        else:
            corrected_words.append(word)

    return " ".join(corrected_words)

def generate_emergency_summary(category: str, transcript: str, address: str, date_str: str, time_str: str) -> str:
    text_lower = transcript.lower()
    
    if any(k in text_lower for k in ["dugo", "nasaktan", "nahulog", "nahimatay", "sugat", "nasaksak"]):
        patient_status = "CRITICAL / URGENT MEDICAL ASSISTANCE"
    elif any(k in text_lower for k in ["nasusunog", "apoy", "usok"]):
        patient_status = "ACTIVE FIRE / IMMEDIATE EVACUATION"
    elif any(k in text_lower for k in ["pulis", "nakawan", "gulo"]):
        patient_status = "LAW ENFORCEMENT DISPATCH"
    else:
        patient_status = "PRIORITY DISPATCH REQUIRED"

    summary = (
        f"• DATE & TIME: {date_str} at {time_str}\n"
        f"• CATEGORY: {category}\n"
        f"• STATUS: {patient_status}\n"
        f"• TRANSCRIPT: \"{transcript}\"\n"
        f"• LOCATION: {address}"
    )
    return summary

incidents_db: List[dict] = []

def transcribe_audio_sync(file_path: str) -> str:
    clean_audio_path = denoise_audio_file(file_path)

    segments, _ = stt_model.transcribe(
        clean_audio_path,
        language="tl",
        task="transcribe",
        beam_size=4,
        best_of=4,
        temperature=0.0,
        vad_filter=False,
        condition_on_previous_text=False,
        initial_prompt="Ito ay emergency report sa Tagalog: kalye, kalsada, kanal, hagdanan, naaksidente, nabangga, kotse, bata, tulong, saklolo, ambulansya, pulis, bumbero."
    )
    
    raw_text = " ".join([segment.text for segment in segments]).strip()
    context_text = normalize_context_phrases(raw_text)
    return spellcheck_tagalog_words(context_text)

@app.post("/api/v1/incidents/report")
async def receive_incident_report(
    audio: Optional[UploadFile] = File(None),
    latitude: str = Form(...),
    longitude: str = Form(...),
    callerName: str = Form(...),
    phoneNumber: str = Form(...),
    address: str = Form(...)
):
    global incidents_db
    start_time = time.time()
    try:
        incident_id = f"INC-{uuid.uuid4().hex[:4].upper()}"
        file_filename = f"{incident_id}_recording.aac"
        file_path = os.path.join(AUDIO_DIR, file_filename)
        
        now = datetime.now()
        date_str = now.strftime("%B %d, %Y")
        time_str = now.strftime("%I:%M:%S %p")
        full_timestamp = f"{date_str} - {time_str}"

        transcription_text = ""

        if audio and audio.filename:
            contents = await audio.read()
            if len(contents) > 100:
                with open(file_path, "wb") as buffer:
                    buffer.write(contents)

                stt_start = time.time()
                transcription_text = await asyncio.to_thread(transcribe_audio_sync, file_path)

        if not transcription_text:
            transcription_text = "May emergency report na ipinadala ang tumawag."

        category = "MEDIKAL / AMBULANCE"
        if bert_model and tokenizer and transcription_text:
            inputs = tokenizer(
                transcription_text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=64
            )
            with torch.inference_mode():
                outputs = bert_model(**inputs)
                predicted_class_id = torch.argmax(outputs.logits, dim=1).item()
                
            category = CATEGORY_MAP.get(predicted_class_id, "MEDIKAL / AMBULANCE")

        summary_report = generate_emergency_summary(category, transcription_text, address, date_str, time_str)
        audio_url = f"http://localhost:8000/static/audio/{file_filename}" if os.path.exists(file_path) else ""

        new_incident = {
            "status": "success",
            "incidentId": incident_id,
            "callerName": callerName,
            "phoneNumber": phoneNumber,
            "address": address,
            "latitude": float(latitude) if latitude else 13.1391,
            "longitude": float(longitude) if longitude else 123.7438,
            "reportDate": date_str,
            "reportTime": time_str,
            "timestamp": full_timestamp,
            "transcription": transcription_text,
            "category": category,
            "summary": summary_report,
            "audioUrl": audio_url,
            "distanceEta": "2.5 km (5 mins)"
        }

        incidents_db.insert(0, new_incident)
        return new_incident

    except Exception as e:
        print(f"Error handling report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/incidents/all")
async def get_all_incidents():
    return incidents_db

@app.get("/api/v1/incidents/latest")
async def get_latest_incident():
    return incidents_db[0] if incidents_db else {}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)