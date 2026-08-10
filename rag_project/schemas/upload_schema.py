from pydantic import BaseModel

class UploadResponse(BaseModel):
    document_id: str
    file_name: str
    status: str
    stored_in_db: str
