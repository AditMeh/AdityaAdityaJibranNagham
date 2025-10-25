import sounddevice as sd
import numpy as np
import io
import wave
import base64
import openai
import os
import time

# Run in your terminal `export BOSON_API_KEY="your_key_here"`
BOSON_API_KEY = os.getenv("BOSON_API_KEY")

client = openai.Client(
    api_key=BOSON_API_KEY,
    base_url="https://hackathon.boson.ai/v1"
)

# === CONFIG ===
SAMPLERATE = 16000        # Lower rate = faster processing
CHANNELS = 1
SILENCE_THRESHOLD = 0.15  # Adjust if too sensitive (0.01–0.05 typical)
MIN_DURATION = 1        # Min seconds of speech before sending
MAX_DURATION = 10         # Hard cap to avoid overly long clips
SILENCE_TIME = 1.5        # Time (s) of silence before cutting

def record_until_silence():
    """Continuously listen and record when speech detected, stop on silence."""
    print("🎧 Listening for speech...")
    audio_buffer = []
    is_recording = False
    silence_start = None
    start_time = None

    with sd.InputStream(samplerate=SAMPLERATE, channels=CHANNELS, dtype=np.float32) as stream:
        while True:
            data, _ = stream.read(int(SAMPLERATE * 0.1))  # process 100 ms chunks
            volume = np.abs(data).mean()

            if not is_recording:
                # Start recording when volume passes threshold
                if volume > SILENCE_THRESHOLD:
                    is_recording = True
                    start_time = time.time()
                    audio_buffer = [data]
                    print("🎙️ Detected speech… recording...")
            else:
                audio_buffer.append(data)
                duration = time.time() - start_time

                # Stop if silent for a while or duration too long
                if volume < SILENCE_THRESHOLD:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > SILENCE_TIME:
                        print("🛑 Silence detected — stopping.")
                        break
                else:
                    silence_start = None

                if duration >= MAX_DURATION:
                    print("⏱️ Max duration reached (10 s) — stopping.")
                    break

        if not is_recording or len(audio_buffer) == 0:
            return None

        audio = np.concatenate(audio_buffer, axis=0)
        duration = len(audio) / SAMPLERATE
        if duration < MIN_DURATION:
            print("⚠️ Clip too short, ignored.")
            return None

        # Convert to WAV (16-bit PCM)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLERATE)
            wf.writeframes((audio * 32767).astype(np.int16).tobytes())

        return base64.b64encode(buffer.getvalue()).decode("utf-8")

def analyze_audio(audio_b64):
    """Send the audio clip to Boson and print the model's response."""
    print("🔊 Sending to Boson model...")
    response = client.chat.completions.create(
        model="higgs-audio-understanding-Hackathon",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Transcribe the audio using correct punctuation."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_b64, "format": "wav"},
                    },
                ],
            },
        ],
        max_completion_tokens=128,
        temperature=0,
    )

    reply = response.choices[0].message.content
    print(f"\n🧠 Model reply:\n{reply}\n" + "-" * 40)
    return reply

def main():
    print("🎤 Boson Voice Listener (Ctrl+C to stop)\n")
    try:
        while True:
            audio_b64 = record_until_silence()
            if audio_b64:
                analyze_audio(audio_b64)
    except KeyboardInterrupt:
        print("\n👋 Exiting gracefully.")

if __name__ == "__main__":
    main()
