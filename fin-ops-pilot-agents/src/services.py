import os
import fitz
import openai
from pathlib import Path
from typing import Dict
import asyncio
from dotenv import load_dotenv
from livekit.api import LiveKitAPI 
from livekit.protocol.sip import CreateSIPParticipantRequest, SIPParticipantInfo
from .config import Settings
from .utils import sliding_window_summarize

load_dotenv()

def process_and_summarize_pdf(file_path: Path, api_key: str) -> Dict[str, str]:
    """
    Reads a PDF file, extracts its text, and generates a summary.

    Args:
        file_path: Path to the PDF file.
        api_key: OpenAI API key.

    Returns:
        A dictionary containing the summary and the path where it was stored.
    """
    pdf_text = ""
    doc = fitz.open(file_path)
    for page in doc:
        pdf_text += page.get_text()
    doc.close()

    summary = sliding_window_summarize(doc=pdf_text, api_key=api_key)

    return summary


async def create_client_details_from_notification(
    notification_message: str, context_dir: Path, policy_schedule_path: Path, client: openai.OpenAI
) -> Path:
    """
    Generates client details by summarizing a policy schedule and appending notification message.

    Args:
        notification_message: The message from the notification.
        context_dir: The directory to store context files.
        policy_schedule_path: Path to the policy schedule document.
        client: An initialized OpenAI client.

    Returns:
        The path to the created client_details.txt file.
    """
    if not policy_schedule_path.exists():
        raise FileNotFoundError("policy_schedule.txt not found.")

    policy_details = policy_schedule_path.read_text(encoding="utf-8")

    prompt = f"Identify Policy owner details, generate response in third person and in one line:\n\n{policy_details}"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert at creating concise, one-line summaries."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=150,
    )
    one_line_summary = response.choices[0].message.content.strip()

    client_details_content = f"{one_line_summary}\n{notification_message}"

    client_details_path = context_dir / "client_details.txt"
    client_details_path.write_text(client_details_content, encoding="utf-8")

    livekit_api = LiveKitAPI(url=os.getenv("LIVEKIT_URL"), api_key=os.getenv("LIVEKIT_API_KEY"), api_secret=os.getenv("LIVEKIT_API_SECRET"))

    request = CreateSIPParticipantRequest(
        sip_trunk_id = "ST_xkDdmL7TYv22",
        sip_call_to = "+917249469755",
        room_name = "my-sip-room",
        participant_identity = "sip-test",
        participant_name = "Test Caller",
        display_name = "Insurance caller",
        krisp_enabled = True,
        wait_until_answered = True
    )
    
    try:
        participant = await livekit_api.sip.create_sip_participant(request)
        print(f"Successfully created {participant}")
    except Exception as e:
        print(f"Error creating SIP participant: {e}")
        # sip_status_code contains the status code from upstream carrier
        print(f"SIP error code: {e.metadata.get('sip_status_code')}")
        # sip_status contains the status message from upstream carrier
        print(f"SIP error message: {e.metadata.get('sip_status')}")
    finally:
        await livekit_api.aclose()

    return client_details_path


def save_summary(summary: str, output_path: Path):
    """Saves the summary text to a file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(summary, encoding="utf-8")