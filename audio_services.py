# core/audio_services.py (FINAL CORRECTED VERSION)

import os
import asyncio
import aiosqlite
import sqlite3 # Keep sync sqlite3 for the sync processor_task
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import Optional # <--- 添加这行缺失的代码
import pyperclip
from faster_whisper import WhisperModel

from config import settings
from core import state
from utils.helpers import Colors, is_primarily_chinese, get_local_time_str, safe_notification

async def setup_whisper_model():
    """[ASYNC] Loads the faster-whisper model from a local folder without blocking."""
    try:
        local_model_path = "./faster-whisper-large-v3-local"
        if not os.path.isdir(local_model_path):
            print(f"❌ {Colors.RED}[Whisper Error] Model folder not found at '{local_model_path}'!{Colors.ENDC}")
            return False

        print(f">> [Async Audio Service] Loading Whisper model in background thread...")
        state.WHISPER_MODEL = await asyncio.to_thread(
            WhisperModel, local_model_path, device="cpu", compute_type="int8"
        )
        print(f"✅ {Colors.GREEN}[Async Audio Service] Whisper model loaded successfully.{Colors.ENDC}")
        return True
    except Exception as e:
        print(f"❌ {Colors.RED}[Whisper Error] Failed to load local model: {e}{Colors.ENDC}")
        return False

async def piper_tts_task(text: str):
    """
    [ASYNC] Generates and plays audio using Piper in a non-blocking way.
    """
    temp_audio_file = None
    try:
        def _run_subprocess():
            # This synchronous function contains all the blocking calls
            print(f"\n{Colors.BLUE}[Audio Service] Generating TTS audio for text (length: {len(text)})...{Colors.ENDC}")
            script_dir = Path("./")
            piper_exe = script_dir / "piper" / "piper"
            if is_primarily_chinese(text):
                model_file, config_file = script_dir / "zh_CN-huayan-medium.onnx", script_dir / "zh_CN-huayan-medium.onnx.json"
            else:
                model_file, config_file = script_dir / "en_US-lessac-medium.onnx", script_dir / "en_US-lessac-medium.onnx.json"
            
            nonlocal temp_audio_file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp_audio:
                temp_audio_file = fp_audio.name

            piper_command = [
                str(piper_exe), "--model", str(model_file),
                "--config", str(config_file), "--output_file", temp_audio_file,
            ]
            
            print("   -> [Thread] Executing Piper command via STDIN...")
            subprocess.run(
                piper_command, check=True, capture_output=True,
                input=text.encode("utf-8"), timeout=180,
            )
            print("   -> [Thread] Audio generated. Starting playback...")
            state.tts_process = subprocess.Popen(["aplay", temp_audio_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            state.tts_process.wait()
            print(f"   -> {Colors.GREEN}[Thread] Playback complete.{Colors.ENDC}")

        await asyncio.to_thread(_run_subprocess)
    except Exception as e:
        print(f"   -> {Colors.RED}An unexpected error occurred during TTS: {e}{Colors.ENDC}")
        if state.main_loop:
            asyncio.run_coroutine_threadsafe(safe_notification("TTS Error", "Failed to generate or play audio."), state.main_loop)
    finally:
        if temp_audio_file and os.path.exists(temp_audio_file):
            os.remove(temp_audio_file)
        state.tts_process = None

# In core/audio_services.py
# REPLACE the _process_single_file AND processor_task functions with this block.

# ==============================================================================
#      【【【 THIS IS THE MAIN FIX BLOCK 】】】
# ==============================================================================
async def _process_single_file(audio_path_str: str, session_id: Optional[str]):
    """
    [ASYNC WORKER] Handles one audio file.
    It now receives the session_id directly to avoid race conditions.
    """
    try:
        audio_path = Path(audio_path_str)
        print(f"\n{Colors.BLUE}>> [Async Processor] Picked up task: {audio_path.name}{Colors.ENDC}")
        await safe_notification("New Transcription", "Processing audio chunk...")

        segments, info = await asyncio.to_thread(state.WHISPER_MODEL.transcribe, audio_path_str)
        full_text = " ".join(segment.text for segment in segments).strip()

        if full_text:
            is_meeting_chunk = audio_path.name.startswith("rec_")
            prefix = "[Meeting] " if is_meeting_chunk else ""
            final_text = f"{prefix}{full_text}"

            print(f"   -> [Processor] Transcription complete (Lang: {info.language}). Saving to DB.")
            
            # Use the session_id that was passed directly with the file.
            async with aiosqlite.connect(str(settings.VOICE_TRANSCRIPTS_DB)) as conn:
                await conn.execute(
                    f"""INSERT INTO {settings.VOICE_TRANSCRIPTS_TABLE_NAME}
                        (timestamp, language_detected, transcribed_text, session_id)
                        VALUES (?, ?, ?, ?)""",
                    (get_local_time_str(), info.language, final_text, session_id),
                )
                await conn.commit()
            print(f"   -> [Processor] ✅ Transcript saved to database with session ID: {str(session_id)[:8]}...")
            
            if not is_meeting_chunk:
                 await asyncio.to_thread(pyperclip.copy, full_text)
                 await safe_notification("Transcription Complete", "Text copied to clipboard.")
                 print(f"   -> {Colors.GREEN}[Clipboard] ✅ Transcribed text copied!{Colors.ENDC}")
            
            # If this was the final chunk of a meeting, signal completion.
            if audio_path.name.startswith("rec_final"):
                # CRITICAL FIX: event.set() is NOT a coroutine. It's a thread-safe sync method.
                # We can and should call it directly.
                state.processing_complete_event.set()
                print(f"   -> {Colors.CYAN}[Processor] Signaled completion for meeting session.{Colors.ENDC}")
        else:
            print(f"   -> [Processor] {Colors.YELLOW}No speech detected.{Colors.ENDC}")

        if not settings.KEEP_AUDIO_FILES and os.path.exists(audio_path_str):
            os.unlink(audio_path_str)

    except Exception as e:
        print(f"❌ {Colors.RED}[Async Processor Error] Failed to process {Path(audio_path_str).name}: {e}{Colors.ENDC}")
        traceback.print_exc()

def processor_task():
    """
    [SYNC WRAPPER] Runs in a dedicated background thread.
    It now gets a tuple (filepath, session_id) from the queue.
    """
    while True:
        try:
            # Get the tuple from the queue
            item = state.transcription_queue.get()
            audio_path_str, session_id = item

            if state.main_loop and state.main_loop.is_running():
                # Pass both arguments to the async worker
                future = asyncio.run_coroutine_threadsafe(
                    _process_single_file(audio_path_str, session_id), state.main_loop
                )
                future.result()
            
            state.transcription_queue.task_done()
        except Exception as e:
            print(f"❌ {Colors.RED}[Audio Processor Thread Error] A critical error occurred: {e}{Colors.ENDC}")
            traceback.print_exc()
# ==============================================================================
#      【【【 END OF FIX BLOCK 】】】
# ==============================================================================
