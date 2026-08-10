"""
POST /upload — user drops one or more files here.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from app.schemas.upload_schema import UploadResponse
from pathlib import Path
import aiofiles
import uuid
from app.graphs.storing_graph import ingestion_app

router = APIRouter()

UPLOAD_DIR = Path("/workspaces/rag/app/data")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


@router.post("", response_model=list[UploadResponse])
async def upload_files(files: list[UploadFile] = File(...)):
    results = []

    for file in files:
        if not file.filename:
            raise HTTPException(status_code=400, detail="File must have a filename")

        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext} ({file.filename})")

        document_id = str(uuid.uuid4())
        file_path = UPLOAD_DIR / f"{file.filename}{ext}"

        try:
            async with aiofiles.open(file_path, "wb") as out_file:
                while chunk := await file.read(1024 * 1024):
                    await out_file.write(chunk)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save {file.filename}: {e}")

        ingestion_app.invoke({
            "chunks" : [],
            "documents" : []
        })

        results.append(
            UploadResponse(
                document_id=document_id,
                file_name=file.filename,
                status="success",
                stored_in_db="success"
            )
        )

    return results