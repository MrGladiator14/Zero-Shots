import logging
import os
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    inference,
    room_io,
    function_tool,
    RunContext
)
from pypdf import PdfReader
from livekit.plugins import (
    openai,
    cartesia,
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env")


class Assistant(Agent):
    def __init__(self) -> None:
        base_instructions = """You are a helpful voice AI assistant For a Insurance company. The user is interacting with you via voice, even if you perceive the conversation as text.
            You eagerly assist users with their questions by providing information from the provided 'Policy Schedule' document.
            Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            You are curious, friendly and professional"""

        client_details_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "context", "client_details.txt"
        )
        try:
            with open(client_details_path, "r") as f:
                client_details = f.read()
            instructions = f"{base_instructions}\n{client_details}"
        except FileNotFoundError:
            instructions = base_instructions
        super().__init__(instructions=instructions)

    @function_tool
    async def get_policy_schedule_details(self, context: RunContext) -> str:
        """Use this tool to get the user's insurance policy schedule details.
        This tool reads the 'Policy Schedule' document and returns its contents as text.
        """
        logger.info("loading policy schedule details")
        policy_schedule_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "context", "policy_schedule.txt"
        )
        try:
            with open(policy_schedule_path, "r") as f:
                policy_schedule_details = f.read()
            return
        except FileNotFoundError:
            return "No policy schedule details found."
    
    # @function_tool
    # async def end_session(self):
    #     """When the user wants to stop talking, use this to close the session."""
    #     await self.session.drain()
    #     await self.session.aclose()
    
    async def on_exit(self):
        await self.session.say("Goodbye!")


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def my_agent(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session = AgentSession(
        # stt=inference.STT(model="assemblyai/universal-streaming", language="en"),
        # llm=inference.LLM(model="openai/gpt-4.1-mini"),
        # tts=inference.TTS(
        #     model="cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
        # ),
        # vad=ctx.proc.userdata["vad"],

        stt=cartesia.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=cartesia.TTS(),
        vad=silero.VAD.load(),

        turn_detection=MultilingualModel(),
        preemptive_generation=True,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=noise_cancellation.BVCTelephony()
                # noise_cancellation=lambda params: noise_cancellation.BVCTelephony()
                # if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                # else noise_cancellation.BVC(),
            ),
        ),
    )

    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
