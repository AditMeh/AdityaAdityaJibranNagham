# main.py
import threading
import queue
import sys
import json
import time

# Add LM folder to Python path
sys.path.append("../jibran")
from lmoutput import * # now you can access SYSTEM_PROMPT and client
import native

from liveAudioTranscribing import record_until_silence, analyze_audio

# === Listener thread: live audio -> transcription -> LM normalization ===
def audio_listener_loop():
    while True:
        audio_b64 = record_until_silence()
        if audio_b64:
            # Get transcription from Boson
            transcription = analyze_audio(audio_b64)  # make analyze_audio return text
            if not transcription:
                continue

            print(f"\n🎤 Transcribed text: {transcription}")

            # Normalize using LM and store edits in queue
            edits = get_lm_output(transcription)
            if edits: 
                print(f"✅ Queued {len(edits)} edits.")

                """Continuously processes items from the edit queue."""
                for edit in edits:
                    print("🖌️ Processing edit:", edit)

            else:
                print("No instructions detected")


# === MAIN ===
if __name__ == "__main__":
    # Start listener thread
    listener_thread = threading.Thread(target=audio_listener_loop, daemon=True)
    listener_thread.start()

    print("🎤 Live audio listener started. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Exiting gracefully.")
