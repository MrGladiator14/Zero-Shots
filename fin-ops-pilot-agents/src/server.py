from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
from typing import Annotated
from fastapi import Depends

from .config import Settings, settings as app_settings
from . import services

# --- Dependency Injection ---

def get_settings() -> Settings:
    return app_settings

def get_openai_client(settings: Annotated[Settings, Depends(get_settings)]) -> openai.OpenAI:
    """Dependency to create and provide an OpenAI client."""
    return openai.OpenAI(api_key=settings.OPENAI_API_KEY)

app = FastAPI()

# CORS Middleware to allow requests from the frontend
origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Notification(BaseModel):
    message: str
    context_document: str

class UploadDocRequest(BaseModel):
    name: str

@app.get("/")
def read_root():
   return {"message": "Hello, World!"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
   return {"item_id": item_id, "q": q}

@app.post("/api/notify")
async def notify(
    notification: Notification,
    settings: Annotated[Settings, Depends(get_settings)],
    client: Annotated[openai.OpenAI, Depends(get_openai_client)],
):
    try:
        policy_schedule_path = settings.CONTEXT_DIR / "policy_schedule.txt"
        client_details_path = await services.create_client_details_from_notification(
            notification_message=notification.message,
            context_dir=settings.CONTEXT_DIR,
            policy_schedule_path=policy_schedule_path,
            client=client,
        )
        return {
            "status": "success",
            "message": "Client details created successfully.",
            "path": str(client_details_path),
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except openai.APIError as e:
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {e.message}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/api/upload_doc")
async def upload_doc(
    settings: Annotated[Settings, Depends(get_settings)],
    request_data: UploadDocRequest,
):
    """
    Receives a file upload, saves it, processes it to generate a summary,
    and stores the summary.
    """
    # Ensure the upload directory exists
    settings.UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)

    file_path = settings.UPLOAD_DIRECTORY / request_data.name
    
    try:
        summary = services.process_and_summarize_pdf(
            file_path=file_path, api_key=settings.OPENAI_API_KEY
        )

        # Store the summary to the specified file
        summary_path = settings.CONTEXT_DIR / "policy_schedule.txt"
        services.save_summary(summary, summary_path)

        return {
            "summary": summary,
            "storage_path": str(summary_path),
        }
    except openai.APIError as e:
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {e.message}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")