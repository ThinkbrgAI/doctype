from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from document_classifier import DocumentClassifier
from config import SecureConfig

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProcessFolderRequest(BaseModel):
    folder_path: str
    batch_size: Optional[int] = 10
    model_version: Optional[str] = 'o1-2024-12-17'  # Default to GPT-4 Vision specific version

class FileStatus(BaseModel):
    filename: str
    status: str
    category: Optional[str] = None
    confidence: Optional[float] = None
    error: Optional[str] = None

class ProcessingStatus(BaseModel):
    processed_count: int
    total_files: int
    files: List[FileStatus]
    errors: List[str]

# Global state for batch processing
processing_state = {
    "is_paused": False,
    "current_batch": [],
    "processed_files": [],
    "errors": []
}

@app.post("/api/process-folder", response_model=ProcessingStatus)
async def process_folder(request: ProcessFolderRequest):
    try:
        # Get API key from secure storage
        config = SecureConfig()
        api_key = config.get_api_key()
        if not api_key:
            raise HTTPException(status_code=401, detail="API key not configured")

        # Initialize classifier with selected model
        classifier = DocumentClassifier(api_key, model_version=request.model_version)
        
        # Get list of PDF files
        files = [f for f in os.listdir(request.folder_path) 
                if f.lower().endswith('.pdf')]
        
        # Update processing state
        processing_state["current_batch"] = files[:request.batch_size]
        processing_state["processed_files"] = []
        processing_state["errors"] = []
        
        return ProcessingStatus(
            processed_count=0,
            total_files=len(files),
            files=[FileStatus(filename=f, status="pending") for f in files],
            errors=[]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process-batch")
async def process_batch():
    try:
        if processing_state["is_paused"]:
            return {"status": "paused"}
            
        # Get next batch of files
        batch = processing_state["current_batch"]
        if not batch:
            return {"status": "completed"}
            
        # Process each file in the batch
        for filename in batch:
            try:
                file_path = os.path.join(request.folder_path, filename)
                category, confidence = classifier.classify_document(file_path)
                
                processing_state["processed_files"].append(
                    FileStatus(
                        filename=filename,
                        status="completed",
                        category=category,
                        confidence=confidence
                    ).dict()  # Convert Pydantic model to dict for JSON response
                )
            except Exception as e:
                processing_state["errors"].append(f"Error processing {filename}: {str(e)}")
                
        # Update current batch
        processing_state["current_batch"] = []
        
        return {
            "status": "processing",
            "processed_files": processing_state["processed_files"],
            "errors": processing_state["errors"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/pause")
async def pause_processing():
    processing_state["is_paused"] = True
    return {"status": "paused"}

@app.post("/api/resume")
async def resume_processing():
    processing_state["is_paused"] = False
    return {"status": "resumed"} 