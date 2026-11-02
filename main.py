import os
import shutil
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ==========================================
# 1. FASTAPI APP INITIALIZATION & CORS FIX
# ==========================================
app = FastAPI(
    title="SAKLOLO Emergency Backend",
    description="Local Edge AI Backend powered by Whisper, SymSpell, and BERT",
    version="2.0.0"
)

# Enable CORS for GitHub Pages, Ngrok, and Android App Requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows GitHub Pages (HTTPS) and mobile requests
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static folder for storing audio uploads
UPLOADS_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")


# ==========================================
# 2. IN-MEMORY INCIDENT DATABASE & SCHEMAS
# ==========================================
class IncidentReport(BaseModel):
    incidentId: str
    callerName: str
    phoneNumber: str
    latitude: float
    longitude: float
    address: str
    transcription: str
    category: str
    summary: str
    reportDate: str
    reportTime: str
    audioUrl: Optional[str] = None


# In-memory storage for active incidents
incidents_db: List[IncidentReport] = []


# ==========================================
# 3. AI PIPELINE STUBS (WHISPER / SYMSPELL / BERT)
# ==========================================
def process_voice_pipeline(file_path: str) -> dict:
    """
    Executes the Speech-to-Text, SymSpell correction, and BERT classification pipeline.
    Replace/connect your loaded local models (faster-whisper, symspell, saklolo_bert_model) here.
    """
    # 1. Transcription (Faster-Whisper)
    # transcript = whisper_model.transcribe(file_path)
    mock_transcription = "May malaking sunog po dito sa barangay, kailangan ng bumbero agad!"

    # 2. Normalization & Typo Correction (SymSpell)
    # corrected_text = symspell.correct(mock_transcription)

    # 3. NLP Classification & Extraction (BERT)
    # category, priority = bert_model.predict(corrected_text)
    mock_category = "FIRE / SUNOG"
    mock_summary = "Emergency Level: CRITICAL\nRequired Response: Bureau of Fire Protection (BFP)\nExtracted Details: Active structural fire reported in residential area."

    return {
        "transcription": mock_transcription,
        "category": mock_category,
        "summary": mock_summary
    }


# ==========================================
# 4. API ENDPOINTS
# ==========================================

@app.get("/")
def read_root():
    return {"system": "SAKLOLO AI Backend", "status": "ONLINE"}


@app.get("/api/v1/incidents/all", response_model=List[IncidentReport])
def get_all_incidents():
    """
    Returns all registered emergency incident reports for the dispatcher dashboard map.
    """
    return incidents_db


@app.post("/api/v1/incidents/report", response_model=IncidentReport, status_code=Status.HTTP_201_CREATED)
async def create_incident_report(
    audio: UploadFile = File(...),
    callerName: str = Form("Anonymous Reporter"),
    phoneNumber: str = Form("N/A"),
    latitude: float = Form(...),
    longitude: float = Form(...),
    address: str = Form("Unknown Location")
):
    """
    Receives voice recording & GPS data from mobile app, runs AI inference, and saves the report.
    """
    try:
        # Save incoming audio file locally
        file_extension = os.path.splitext(audio.filename)[1] or ".aac"
        unique_filename = f"{uuid.uuid4().hex}{file_extension}"
        saved_file_path = os.path.join(UPLOADS_DIR, unique_filename)

        with open(saved_file_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)

        # Run AI Processing Pipeline
        ai_results = process_voice_pipeline(saved_file_path)

        # Timestamps
        now = datetime.now()
        report_date = now.strftime("%B %d, %Y")
        report_time = now.strftime("%I:%M:%S %p")
        incident_id = f"SAK-{uuid.uuid4().hex[:6].upper()}"

        # Construct Public Audio URL (served via Ngrok)
        audio_url = f"https://cultural-bonanza-flagstone.ngrok-free.dev/uploads/{unique_filename}"

        # Build Incident Model
        new_incident = IncidentReport(
            incidentId=incident_id,
            callerName=callerName,
            phoneNumber=phoneNumber,
            latitude=latitude,
            longitude=longitude,
            address=address,
            transcription=ai_results["transcription"],
            category=ai_results["category"],
            summary=ai_results["summary"],
            reportDate=report_date,
            reportTime=report_time,
            audioUrl=audio_url
        )

        # Prepend to database (newest reports first)
        incidents_db.insert(0, new_incident)
        print(f"[+] New Emergency Report Created: {incident_id} | Category: {ai_results['category']}")

        return new_incident

    except Exception as e:
        print(f"[-] Error processing report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


# ==========================================
# 5. SERVER LAUNCH
# ==========================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)