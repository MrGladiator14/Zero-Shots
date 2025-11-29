import os
import uuid
import csv
import asyncio
from pathlib import Path
from typing import Optional,Annotated
from datetime import datetime
from fastapi import Depends
from pydantic import BaseModel
import openai
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate


from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="FinOps Copilot API")

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Notification(BaseModel):
    message: str
    context_document: str

class UploadDocRequest(BaseModel):
    name: str

# Storage paths
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# In-memory job tracking
jobs: dict[str, dict] = {}


# --- Models ---
class NotifyRequest(BaseModel):
    message: str
    context_document: str


class WorkflowRequest(BaseModel):
    title: str
    prompt: str
    parallelCount: int = 1


def get_insurer_config(title: str) -> dict:
    """Get credentials and input variables based on insurer type."""
    title_lower = title.lower()
    
    if "royal" in title_lower or "sundaram" in title_lower:
        return {
            "username": os.getenv("ROYAL_SUNDARAM_USERNAME"),
            "password": os.getenv("ROYAL_SUNDARAM_PASSWORD"),
            "input_variables": ["username", "password", "registration_number", "first_name", "last_name", "whatsapp_number"],
        }
    elif "reliance" in title_lower:
        return {
            "username": os.getenv("RELIANCE_USERNAME"),
            "password": os.getenv("RELIANCE_PASSWORD"),
            "input_variables": ["username", "password", "registration_number"],
        }
    else:
        # Default fallback
        return {
            "username": os.getenv("DEFAULT_USERNAME", ""),
            "password": os.getenv("DEFAULT_PASSWORD", ""),
            "input_variables": ["username", "password", "registration_number", "today_date"],
        }


def get_records_from_csv(csv_path: str, insurer_type: str) -> list[dict]:
    """Parse CSV and extract records based on insurer type."""
    records = []
    insurer_type_lower = insurer_type.lower()
    
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "royal" in insurer_type_lower or "sundaram" in insurer_type_lower:
                record = {
                    "registration_number": row.get("registration_number", ""),
                    "first_name": row.get("first_name", ""),
                    "last_name": row.get("last_name", ""),
                    "whatsapp_number": row.get("whatsapp_number", ""),
                }
            elif "reliance" in insurer_type_lower:
                record = {
                    "registration_number": row.get("registration_number", ""),
                }
            else:
                record = {
                    "registration_number": row.get("registration_number", ""),
                }
            records.append(record)
    return records


def format_prompts_for_records(
    prompt_template: PromptTemplate,
    records: list[dict],
    universal_values: dict,
) -> list[dict]:
    """
    Format prompt template for each record, merging universal values.
    
    Returns list of dicts with 'identifier' and 'formatted_prompt'.
    """
    formatted = []
    for i, record in enumerate(records):
        # Merge universal values with record-specific values
        all_values = {**universal_values, **record}
        
        # Format the prompt
        formatted_prompt = prompt_template.format(**all_values)
        
        # Create identifier
        identifier = f"{record.get('first_name', 'record')}_{record.get('registration_number', str(i))}"
        
        formatted.append({
            "identifier": identifier,
            "formatted_prompt": formatted_prompt,
            "record": record,
        })
    return formatted


# --- Endpoints ---
@app.get("/")
def read_root():
    return {"message": "FinOps Copilot API"}


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


async def run_browser_agent(job_id: str, formatted_prompts: list[dict]):
    """UploadDocRequest
    Background task to run browser_use agents for each formatted prompt.
    
    Args:
        job_id: Unique job identifier
        formatted_prompts: List of dicts with 'identifier' and 'formatted_prompt'
    """
    from browser_use import Agent, Browser
    from browser_use import ChatBrowserUse
    from collections import Counter
    from aioconsole import ainput

    def is_agent_stuck(agent: Agent, identifier: str, repeat_threshold: int = 4, step_threshold: int = 4) -> bool:
        history = agent.history.history
        if len(history) < repeat_threshold:
            return False

        recent_actions = []
        for item in history[-step_threshold:]:
            if item.model_output and item.model_output.action:
                for action in item.model_output.action:
                    action_data = action.model_dump(exclude_unset=True)
                    action_name = next(iter(action_data.keys()), None)
                    if action_name:
                        params = action_data.get(action_name, {})
                        if isinstance(params, dict):
                            sig = f"{action_name}:{params.get('index', params.get('coordinate_x', ''))}"
                        else:
                            sig = action_name
                        recent_actions.append(sig)

        action_counts = Counter(recent_actions)
        for action_sig, count in action_counts.items():
            if count >= repeat_threshold:
                print(f"    [{identifier}] [Stuck Detection] Action '{action_sig}' repeated {count} times")
                return True

        recent_goals = []
        for item in history[-step_threshold:]:
            if item.model_output and hasattr(item.model_output, 'current_state'):
                current_state = item.model_output.current_state
                if hasattr(current_state, 'next_goal') and current_state.next_goal:
                    recent_goals.append(current_state.next_goal)

        if len(recent_goals) >= step_threshold and len(set(recent_goals)) == 1:
            print(f"    [{identifier}] [Stuck Detection] Same goal for {step_threshold} consecutive steps")
            return True

        return False

    async def on_step_end_with_stuck_detection(agent: Agent, identifier: str):
        if is_agent_stuck(agent, identifier):
            print(f"\n⚠️  [{identifier}] AGENT APPEARS STUCK - waiting for manual intervention...")
            await ainput("Press Enter after fixing the issue to resume agent execution...")
            print(f"\n▶️  [{identifier}] Resuming agent execution...\n")

    async def run_single_agent(profile_id: int, identifier: str, task: str):
        browser = Browser(
            executable_path=os.getenv("CHROMIUM_PATH", "/usr/bin/chromium"),
            user_data_dir=f"./browser_profiles_{profile_id}",
            profile_directory="Default",
            keep_alive=True,
        )
        agent = Agent(
            task=task,
            browser=browser,
            llm=ChatBrowserUse(api_key=os.getenv("BU_API_KEY")),
            use_vision=True,
            use_thinking=True,
        )

        async def on_step_end_callback(agent: Agent):
            await on_step_end_with_stuck_detection(agent, identifier)

        history = await agent.run(on_step_end=on_step_end_callback)
        return identifier, history.final_result()

    jobs[job_id]["status"] = "running"

    try:
        tasks = [
            run_single_agent(i, item["identifier"], item["formatted_prompt"])
            for i, item in enumerate(formatted_prompts)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["results"] = [
            {"identifier": r[0], "result": str(r[1])} if not isinstance(r, Exception) else {"error": str(r)}
            for r in results
        ]
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


@app.post("/api/execute_workflow")
async def execute_workflow(
    background_tasks: BackgroundTasks,
    title: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    parallelCount: Optional[int] = Form(1),
    csvFile: Optional[UploadFile] = File(None),
):
    """Launch browser_use agents to execute automation tasks."""
    if title is None or prompt is None:
        raise HTTPException(status_code=400, detail="title and prompt are required")

    job_id = str(uuid.uuid4())

    # Get insurer-specific config
    insurer_config = get_insurer_config(title)
    
    # Universal values shared across all records
    universal_values = {
        "username": insurer_config["username"],
        "password": insurer_config["password"],
        "today_date": datetime.now().strftime("%Y-%m-%d"),
    }

    # Create prompt template
    insurance_prompt_template = PromptTemplate(
        input_variables=insurer_config["input_variables"],
        template=prompt,
    )

    # Store CSV and parse records if provided
    csv_path = None
    records = []
    
    if csvFile:
        csv_path = UPLOAD_DIR / f"{job_id}_{csvFile.filename}"
        content = await csvFile.read()
        with open(csv_path, "wb") as f:
            f.write(content)
        # Now read the saved CSV
        records = get_records_from_csv(str(csv_path), title)
    else:
        # No CSV - create a single default record based on parallelCount
        for i in range(parallelCount):
            records.append({"registration_number": f"default_{i}"})

    # Format prompts for each record
    formatted_prompts = format_prompts_for_records(
        insurance_prompt_template,
        records,
        universal_values,
    )

    jobs[job_id] = {
        "id": job_id,
        "title": title,
        "status": "queued",
        "record_count": len(formatted_prompts),
        "csv_path": str(csv_path) if csv_path else None,
    }

    background_tasks.add_task(run_browser_agent, job_id, formatted_prompts)

    return {
        "success": True,
        "job_id": job_id,
        "message": f"Started {len(formatted_prompts)} agent instances for '{title}'",
    }


@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a workflow job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]
