#!/usr/bin/env python3
"""
Terminal controller for image viewer
Connects to the Node.js server via WebSocket and allows terminal control
"""

import websocket
import json
import sys
import threading
import time
import os
from PIL import Image
import sounddevice as sd
import numpy as np
import io
import wave
import base64
import openai
import re
import anthropic
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Native image editing functions
def vflip(image: Image.Image) -> Image.Image:
    """Vertically flips the image."""
    return image.transpose(Image.FLIP_TOP_BOTTOM)

def hflip(image: Image.Image) -> Image.Image:
    """Horizontally flips the image."""
    return image.transpose(Image.FLIP_LEFT_RIGHT)

def grayscale(image: Image.Image) -> Image.Image:
    """Converts image to grayscale."""
    return image.convert('L').convert('RGB')

def rotate(image: Image.Image, degrees: float) -> Image.Image:
    """Rotates the image by a certain number of degrees."""
    # PIL rotates counter-clockwise, so negate degrees for clockwise rotation
    return image.rotate(-degrees, expand=True)

def resize(image: Image.Image, size: tuple) -> Image.Image:
    """Resizes the image to the specified size."""
    return image.resize(size, Image.Resampling.LANCZOS)

def saturation(image: Image.Image, factor: float) -> Image.Image:
    """Adjusts image saturation."""
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(factor)

def sharpness(image: Image.Image, factor: float) -> Image.Image:
    """Adjusts image sharpness."""
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(factor)

def get_lm_output(prompt):
    """Get the output from the LM, returns a JSON array of objects with editing instructions."""
    client = anthropic.Anthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    SYSTEM_PROMPT = f"""You are an intent normalizer for voice-to-image editing.

    Task:
    Given a natural-language instruction, output a JSON array of objects with this exact schema in this order:

    native: boolean

    action: string (present only if native=true), formatted exactly as one of:
    "(resize, (m,n))" where m,n are integers (pixels)
    "(rotate, deg)" where deg ∈ {0, 90, 180, 270} (normalize words like "ninety" → 90). 
    "(grayscale, )"
    "(vflip, )"
    "(hflip, )"
    "(sharpness, s)" where s is a number
    "(saturation, x)" where x is a number

    generative: boolean

    prompt: string (present only if generative=true), a concise diffusion prompt describing the edit for that step

    images: List[str], a list of the images the text asks to use. Leave as Empty if no image is referenced.

    Native vs Generative:

    Native (and only these): resize, rotate, grayscale, vflip, hflip, sharpness, saturation.

    Everything else is Generative: add/remove/replace objects, recolor specific objects or backgrounds, style/material changes, relighting, segmentation/inpainting, background swaps, etc.

    Rules:

    Split multi-step instructions into atomic steps in the spoken order.
    
    For rotate, only include the degree if it is one of the following: {{0, 90, 180, 270}}. Otherwise, ignore the rotate command completely and do not include it in the JSON array.

    For sharpness and saturation, if a percentage is given, use the scale factor corresponding to the percentage. 
    For example, if the instruction is 'increase sharpness by twenty percent' then the constant for sharpness should be 1.2, if the instruction is 'make the saturation 120%' then multiply the saturation by 1.2'
    If we say 'decrease the sharpness by twenty percent' then the constant for sharpness should be 0.8, if we say 'make the saturation 80%' then multiply the saturation by 0.8

    For native steps: native=true, generative=false, fill action, prompt="".

    For generative steps: generative=true, native=false, action="", fill prompt (short, specific, preserve scene unless told otherwise).

    For the images array, do not add a file extension.

    Normalize numbers/units (e.g., "ninety degrees" → 90; "1920 by 1080" → (1920,1080)).

    Remember: Only the seven native actions listed are native. Everything else is generative. Output JSON only and follow the schema order."""

    message = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )
    raw_output = message.content[0].text
    cleaned_json_string = raw_output.strip().removeprefix("```json").strip().removesuffix("```").strip() if raw_output else None
    try:
        json_array = json.loads(cleaned_json_string) if cleaned_json_string else None
    except Exception as e:
        # print(cleaned_json_string)
        # print(f"Error: Invalid JSON string: {cleaned_json_string}")
        # print(f"Error: {e}")
        return None
    return json_array

def get_revert_steps(prompt):
    """Parse revert steps from natural language using LM. Returns the number of steps as an integer."""
    client = anthropic.Anthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    SYSTEM_PROMPT = """You are a number extractor for revert commands.

    Task:
    Given a natural-language instruction about reverting/undoing, extract ONLY the number of steps to revert.

    Examples:
    - "revert back 4 steps" → 4
    - "go back 3 times" → 3
    - "undo 2 steps" → 2
    - "revert 5 steps back" → 5
    - "go back 1 step" → 1
    - "undo 10 times" → 10
    - "revert back" → 1 (default to 1 if no number specified)
    - "undo" → 1 (default to 1 if no number specified)

    Rules:
    - Extract ONLY the numerical digit
    - If no number is specified, return 1
    - If multiple numbers are mentioned, use the first one
    - Normalize words like "one", "two", "three" to digits 1, 2, 3
    - Return ONLY the number, no other text or explanation

    Output format: Just the number as a string (e.g., "4", "1", "10")"""

    message = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=10,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )
    raw_output = message.content[0].text.strip()
    
    try:
        # Extract just the number
        number = int(raw_output)
        return number
    except ValueError:
        # If parsing fails, default to 1
        print(f"⚠️ Could not parse revert steps from '{prompt}', defaulting to 1")
        return 1

def process_generative_edit(image_path, prompt, gpu_server_url="http://localhost:6000/generate"):
    """Process a generative edit using the GPU server."""
    try:
        print(f"🎨 Processing generative edit: '{prompt}'")
        print("⏳ This may take 20-30 seconds...")
        
        with open(image_path, "rb") as image_file:
            files = {
                "image": image_file,
            }
            data = {
                "prompt": prompt,
            }
            
            # Make request with timeout
            response = requests.post(gpu_server_url, files=files, data=data, timeout=60)
            
            if response.status_code == 200:
                # Save the generated image back to the original path
                with open(image_path, "wb") as out_file:
                    print(f"✓ Writing generated image to {image_path}")
                    
                    with open("thingy.png", "wb") as f:
                        f.write(response.content)
                    
                    out_file.write(response.content)
                print(f"✓ Generative edit completed: '{prompt}' → {image_path}")
                return True
            else:
                print(f"✗ Generative edit failed with status {response.status_code}: {response.text}")
                return False
                
    except requests.exceptions.Timeout:
        print("✗ Generative edit timed out (60s)")
        return False
    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to GPU server. Make sure it's running on localhost:6000")
        return False
    except Exception as e:
        print(f"✗ Generative edit error: {e}")
        return False

class ImageViewerController:
    def __init__(self, server_url="ws://localhost:3000"):
        self.server_url = server_url
        self.ws = None
        self.connected = False
        self.running = True
        
        # Voice control configuration
        self.voice_enabled = False
        self.processing_edit = False  # Flag to prevent new voice commands during editing
        self.processing_start_time = None  # Track when processing started
        self.boson_api_key = "bai-AX1rxmAyXUSCH-w_mo0udvxL_m47OTzYVJLfdYd6oluOYYnR"
        if self.boson_api_key:
            self.boson_client = openai.Client(
                api_key=self.boson_api_key,
                base_url="https://hackathon.boson.ai/v1"
            )
        else:
            self.boson_client = None
            print("⚠️  BOSON_API_KEY not set. Voice control disabled.")
        
        # Audio configuration
        self.SAMPLERATE = 16000
        self.CHANNELS = 1
        self.SILENCE_THRESHOLD = 0.053
        self.MIN_DURATION = 1
        self.MAX_DURATION = 10
        self.SILENCE_TIME = 1.5
        
        # Permanent backup system
        self.image_undo_counts = {}  # Track undo count per image
        
    def on_message(self, ws, message):
        """Handle messages from the server"""
        try:
            data = json.loads(message)
            if data['type'] == 'connected':
                print(f"✓ {data['message']}")
                self.connected = True
            elif data['type'] == 'success':
                print(f"✓ {data['message']}")
            elif data['type'] == 'error':
                print(f"✗ {data['message']}")
            elif data['type'] == 'image_list':
                images = data['images']
                if images:
                    print("Available images:")
                    for image in images:
                        print(f"  - {image}")
                else:
                    print("No images found in the directory")
            elif data['type'] == 'current_image':
                if data['image']:
                    print(f"Currently selected: {data['image']}")
                    # Track current image name for undo status
                    self._current_image_name = data['image']
                    # If we're in undo mode, process the image
                    if hasattr(self, '_undo_pending') and self._undo_pending:
                        self._undo_pending = False
                        self.process_image_undo(data['image'])
                    # If we're in multi-step undo mode, process the multi-step undo
                    elif hasattr(self, '_multi_undo_pending') and self._multi_undo_pending:
                        self._multi_undo_pending = False
                        self.process_multi_step_undo(data['image'], self._pending_undo_steps)
                    # If we're in image editing mode, process the edits
                    elif hasattr(self, '_editing_pending') and self._editing_pending:
                        self._editing_pending = False
                        self.process_image_edits(data['image'], self._pending_edits)
                else:
                    message = data.get('message', 'No image currently selected')
                    print(f"No image currently selected: {message}")
                    # Clear current image name
                    self._current_image_name = None
                    # If we're in editing mode but no image, clear the processing flag
                    if hasattr(self, '_editing_pending') and self._editing_pending:
                        self._editing_pending = False
                        self.processing_edit = False
                        self.processing_start_time = None
                        print("⚠️ Cannot edit - no image selected")
        except json.JSONDecodeError:
            print(f"✗ Invalid response from server: {message}")
    
    def on_error(self, ws, error):
        """Handle WebSocket errors"""
        print(f"✗ Connection error: {error}")
        self.connected = False
    
    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close"""
        print("✗ Disconnected from server")
        self.connected = False
    
    def on_open(self, ws):
        """Handle WebSocket open"""
        print("🔌 Connecting to image viewer server...")
        # Send terminal connection message
        ws.send(json.dumps({
            "type": "terminal_connect"
        }))
    
    def connect(self):
        """Connect to the WebSocket server"""
        try:
            self.ws = websocket.WebSocketApp(
                self.server_url,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close
            )
            
            # Run WebSocket in a separate thread
            wst = threading.Thread(target=self.ws.run_forever)
            wst.daemon = True
            wst.start()
            
            # Wait for connection
            timeout = 5
            while not self.connected and timeout > 0:
                time.sleep(0.1)
                timeout -= 0.1
            
            if not self.connected:
                print("✗ Failed to connect to server")
                print("Make sure the Node.js server is running (npm start)")
                return False
            
            return True
            
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            return False
    
    def send_command(self, command):
        """Send a command to the server"""
        if not self.connected or not self.ws:
            print("✗ Not connected to server")
            return False
        
        try:
            self.ws.send(json.dumps(command))
            return True
        except Exception as e:
            print(f"✗ Failed to send command: {e}")
            return False
    
    def select_image(self, image_name):
        """Select an image in the browser"""
        return self.send_command({
            "type": "select_image",
            "image": image_name
        })
    
    def list_images(self):
        """List available images"""
        return self.send_command({
            "type": "list_images"
        })
    
    def get_current_image(self):
        """Get the currently selected image"""
        return self.send_command({
            "type": "get_current_image"
        })
    
    def show_undo_status(self, image_name):
        """Show the current undo status for an image"""
        try:
            test_images_path = './test_images'
            base_name = os.path.splitext(image_name)[0]
            ext = os.path.splitext(image_name)[1]
            
            # Find all existing backup files
            backup_files = []
            highest_number = 0
            
            if os.path.exists(test_images_path):
                for file in os.listdir(test_images_path):
                    if file.startswith(f"{base_name}_old") and file.endswith(ext):
                        try:
                            # Extract number from filename like "image_old1.png" -> 1
                            number_part = file[len(f"{base_name}_old"):-len(ext)]
                            if number_part.isdigit():
                                number = int(number_part)
                                backup_files.append((number, file))
                                highest_number = max(highest_number, number)
                        except ValueError:
                            continue
            
            print(f"📊 Undo status for '{image_name}':")
            print(f"  - Total backups available: {highest_number}")
            
            if highest_number > 0:
                print(f"  - Most recent backup: {base_name}_old{highest_number}{ext}")
                print(f"  - You can undo {highest_number} time{'s' if highest_number != 1 else ''}")
                print("  - All backups are kept permanently")
                
                # Show available backups
                if len(backup_files) > 0:
                    backup_files.sort()
                    print("  - Available backups:")
                    for number, filename in backup_files:
                        print(f"    • {filename}")
            else:
                print("  - No backups available (make edits to create backups)")
                print("  - Backups will be named: image_old1, image_old2, image_old3, etc.")
                
        except Exception as e:
            print(f"✗ Error checking undo status: {e}")
    
    def undo_image(self):
        """Restore image from most recent backup"""
        # Set flag to process undo when we get current image
        self._undo_pending = True
        # First get the current image
        if not self.send_command({"type": "get_current_image"}):
            return False
        
        # Wait for response
        time.sleep(0.1)
        return True
    
    def undo_multiple_steps(self, steps):
        """Undo multiple steps at once"""
        # Set flag to process multi-step undo when we get current image
        self._multi_undo_pending = True
        self._pending_undo_steps = steps
        # First get the current image
        if not self.send_command({"type": "get_current_image"}):
            return False
        
        # Wait for response
        time.sleep(0.1)
        return True
    
    def process_image_undo(self, image_name):
        """Process the image undo using most recent backup (highest numbered _old file) and delete it"""
        try:
            test_images_path = './test_images'
            image_path = os.path.join(test_images_path, image_name)
            
            base_name = os.path.splitext(image_name)[0]
            ext = os.path.splitext(image_name)[1]
            
            # Find all existing backup files and get the highest number
            highest_number = 0
            backup_files = []
            
            for file in os.listdir(test_images_path):
                if file.startswith(f"{base_name}_old") and file.endswith(ext):
                    try:
                        # Extract number from filename like "image_old1.png" -> 1
                        number_part = file[len(f"{base_name}_old"):-len(ext)]
                        if number_part.isdigit():
                            number = int(number_part)
                            backup_files.append((number, file))
                            highest_number = max(highest_number, number)
                    except ValueError:
                        continue
            
            if highest_number == 0:
                print(f"✗ No backups available for '{image_name}'")
                print("You can only undo after making edits to an image")
                return False
            
            # Use the highest numbered backup (most recent)
            most_recent_backup = os.path.join(test_images_path, f"{base_name}_old{highest_number}{ext}")
            
            if not os.path.exists(most_recent_backup):
                print(f"✗ Backup file not found: {base_name}_old{highest_number}{ext}")
                return False
            
            # Restore from most recent backup
            with Image.open(most_recent_backup) as backup_img:
                backup_img.save(image_path)
                print(f"✓ Restored image from backup: {base_name}_old{highest_number}{ext}")
            
            # Delete the used backup
            os.remove(most_recent_backup)
            print(f"✓ Deleted backup: {base_name}_old{highest_number}{ext}")
            
            # Update undo count to reflect remaining backups
            self.image_undo_counts[image_name] = highest_number - 1
            
            # Show remaining undo count
            remaining = self.image_undo_counts[image_name]
            if remaining > 0:
                print(f"📊 Remaining undos: {remaining}")
                print(f"📊 Next backup to restore: {base_name}_old{remaining}{ext}")
            else:
                print("📊 No more undos available for this image")
            
            return True
                
        except Exception as e:
            print(f"✗ Error undoing image: {e}")
            return False
    
    def process_multi_step_undo(self, image_name, steps):
        """Process multi-step undo: restore from target backup and delete all backups from target to highest"""
        try:
            test_images_path = './test_images'
            image_path = os.path.join(test_images_path, image_name)
            
            base_name = os.path.splitext(image_name)[0]
            ext = os.path.splitext(image_name)[1]
            
            # Find all existing backup files and get the highest number
            highest_number = 0
            backup_files = []
            
            for file in os.listdir(test_images_path):
                if file.startswith(f"{base_name}_old") and file.endswith(ext):
                    try:
                        # Extract number from filename like "image_old1.png" -> 1
                        number_part = file[len(f"{base_name}_old"):-len(ext)]
                        if number_part.isdigit():
                            number = int(number_part)
                            backup_files.append((number, file))
                            highest_number = max(highest_number, number)
                    except ValueError:
                        continue
            
            if highest_number == 0:
                print(f"✗ No backups available for '{image_name}'")
                print("You can only undo after making edits to an image")
                return False
            
            # Calculate target backup number
            target_number = highest_number - steps + 1
            
            if target_number < 1:
                print(f"✗ Cannot undo {steps} steps - only {highest_number} backups available")
                print(f"Maximum undo: {highest_number} steps")
                return False
            
            # Check if target backup exists
            target_backup = os.path.join(test_images_path, f"{base_name}_old{target_number}{ext}")
            if not os.path.exists(target_backup):
                print(f"✗ Target backup not found: {base_name}_old{target_number}{ext}")
                return False
            
            # Restore from target backup
            with Image.open(target_backup) as backup_img:
                backup_img.save(image_path)
                print(f"✓ Restored image from backup: {base_name}_old{target_number}{ext}")
            
            # Delete all backups from target to highest (inclusive)
            deleted_count = 0
            for i in range(target_number, highest_number + 1):
                backup_to_delete = os.path.join(test_images_path, f"{base_name}_old{i}{ext}")
                if os.path.exists(backup_to_delete):
                    os.remove(backup_to_delete)
                    deleted_count += 1
                    print(f"✓ Deleted backup: {base_name}_old{i}{ext}")
            
            # Update undo count to reflect remaining backups
            self.image_undo_counts[image_name] = target_number - 1
            
            # Show remaining undo count
            remaining = self.image_undo_counts[image_name]
            print(f"📊 Multi-step undo completed: {steps} steps")
            print(f"📊 Deleted {deleted_count} backup files")
            if remaining > 0:
                print(f"📊 Remaining undos: {remaining}")
                print(f"📊 Next backup to restore: {base_name}_old{remaining}{ext}")
            else:
                print("📊 No more undos available for this image")
            
            return True
                
        except Exception as e:
            print(f"✗ Error processing multi-step undo: {e}")
            return False
    
    def record_until_silence(self):
        """Continuously listen and record when speech detected, stop on silence."""
        if not self.boson_client:
            print("✗ Voice control not available. Set BOSON_API_KEY environment variable.")
            return None
            
        print("🎧 Listening for speech... (Press Ctrl+C to cancel)")
        audio_buffer = []
        is_recording = False
        silence_start = None
        start_time = None
        listen_start = time.time()
        MAX_LISTEN_TIME = 30  # Maximum time to listen for speech (30 seconds)

        try:
            with sd.InputStream(samplerate=self.SAMPLERATE, channels=self.CHANNELS, dtype=np.float32) as stream:
                while True:
                    # Check if we've been listening too long
                    if time.time() - listen_start > MAX_LISTEN_TIME:
                        print("⏱️ Listening timeout (30s) — stopping.")
                        break
                        
                    data, _ = stream.read(int(self.SAMPLERATE * 0.1))  # process 100 ms chunks
                    volume = np.abs(data).mean()

                    if not is_recording:
                        # Start recording when volume passes threshold
                        if volume > self.SILENCE_THRESHOLD:
                            is_recording = True
                            start_time = time.time()
                            audio_buffer = [data]
                            print("🎙️ Detected speech… recording...")
                    else:
                        audio_buffer.append(data)
                        duration = time.time() - start_time

                        # Stop if silent for a while or duration too long
                        if volume < self.SILENCE_THRESHOLD:
                            if silence_start is None:
                                silence_start = time.time()
                            elif time.time() - silence_start > self.SILENCE_TIME:
                                print("🛑 Silence detected — stopping.")
                                break
                        else:
                            silence_start = None

                        if duration >= self.MAX_DURATION:
                            print("⏱️ Max duration reached (10 s) — stopping.")
                            break

            if not is_recording or len(audio_buffer) == 0:
                print("⚠️ No speech detected")
                return None

            audio = np.concatenate(audio_buffer, axis=0)
            duration = len(audio) / self.SAMPLERATE
            if duration < self.MIN_DURATION:
                print("⚠️ Clip too short, ignored.")
                return None

            # Convert to WAV (16-bit PCM)
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(self.SAMPLERATE)
                wf.writeframes((audio * 32767).astype(np.int16).tobytes())

            return base64.b64encode(buffer.getvalue()).decode("utf-8")
            
        except KeyboardInterrupt:
            print("\n🛑 Recording cancelled by user")
            return None
        except Exception as e:
            print(f"✗ Recording error: {e}")
            return None
    
    def transcribe_audio(self, audio_b64):
        """Send the audio clip to Boson and return the transcribed text."""
        if not self.boson_client:
            return None
            
        print("🔊 Sending to Boson model...")
        try:
            # Add timeout to prevent hanging
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("API call timed out")
            
            # Set 15 second timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(15)
            
            response = self.boson_client.chat.completions.create(
                model="higgs-audio-understanding-Hackathon",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Transcribe the audio using correct punctuation. Only return the transcribed text, no additional commentary."},
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
            
            # Cancel timeout
            signal.alarm(0)

            reply = response.choices[0].message.content.strip()
            # Strip all punctuation for more reliable voice commands
            import string
            reply = reply.translate(str.maketrans('', '', string.punctuation)).strip()
            print(f"🎤 Heard: '{reply}'")
            return reply
        except TimeoutError:
            print("✗ API call timed out (15s)")
            return None
        except Exception as e:
            print(f"✗ Transcription error: {e}")
            return None
    
    def process_voice_command(self, command_text):
        """Process a voice command and execute the appropriate action."""
        if not command_text:
            return False
            
        command = command_text.lower().strip()
        print(f"🎯 Processing voice command: '{command}'")
        
        # Send voice command to web interface
        self.send_command({
            "type": "voice_command",
            "command": command_text,
            "timestamp": time.time()
        })
        
        # Map voice commands to actions
        if command in ['list', 'list images', 'show images', 'ls']:
            return self.list_images()
        elif command in ['current', 'current image', 'what image', 'show current']:
            return self.get_current_image()
        elif 'revert' in command or command in ['restore', 'go back']:
            # Check if it's a multi-step revert command
            if any(word in command for word in ['steps', 'times', 'back']):
                # Use LM to parse the number of steps
                steps = get_revert_steps(command)
                print(f"🎯 Parsed revert command: '{command}' → {steps} steps")
                return self.undo_multiple_steps(steps)
            else:
                # Single step revert
                return self.undo_image()
        elif 'undo status' in command or command in ['status', 'undo count']:
            # Show undo status for current image
            if hasattr(self, '_current_image_name') and self._current_image_name:
                self.show_undo_status(self._current_image_name)
                return True
            else:
                print("✗ No image currently selected")
                return False
        elif command.startswith('open ') or command.startswith('select '):
            # Extract image name from "open filename" or "select filename"
            parts = command.split(' ', 1)
            if len(parts) > 1:
                search_term = parts[1].strip()
                return self.find_and_select_image(search_term)
            else:
                print("✗ Please specify an image name")
                return False
        elif command in ['help', 'commands', 'what can i say']:
            self.show_voice_help()
            return True
        elif command in ['quit', 'exit', 'stop', 'goodbye']:
            print("👋 Goodbye!")
            self.running = False
            return True
        elif 'kill' in command or command in ['force quit', 'terminate', 'die']:
            print("💀 Force killing controller...")
            self.force_kill()
            return True
        else:
            # Try to process as image editing command
            print(f"🎨 Processing as image editing command: '{command}'")
            return self.process_image_editing_command(command)
    
    def show_voice_help(self):
        """Show available voice commands."""
        print("\n🎤 Available Voice Commands:")
        print("  'list' or 'list images'     - List available images")
        print("  'current' or 'current image' - Show currently selected image")
        print("  'revert' or 'revert image'   - Restore image from backup (unlimited)")
        print("  'revert X steps' or 'go back X times' - Multi-step revert")
        print("  'status' or 'undo status'    - Show undo status for current image")
        print("  'open [filename]'            - Select an image")
        print("  'help' or 'commands'         - Show this help")
        print("  'quit' or 'exit'             - Exit the controller")
        print("  'kill' or 'force quit'       - Force kill the controller")
        print()
        print("🎨 Image Editing Commands:")
        print("  Say any natural language editing instruction like:")
        print("  'rotate the image 90 degrees'")
        print("  'make it grayscale'")
        print("  'flip it vertically'")
        print("  'increase saturation by 1.5'")
        print("  'resize to 800 by 600'")
        print()
        print("📊 Smart Backup System:")
        print("  - Each edit creates a backup (image_old1, image_old2, etc.)")
        print("  - Use 'revert' to undo the most recent edit (deletes that backup)")
        print("  - New edits reuse deleted backup numbers")
        print("  - Highest number = most recent backup")
        print("  - Use 'status' to see all available backups")
        print()
    
    def toggle_voice_control(self):
        """Toggle voice control on/off."""
        if not self.boson_client:
            print("✗ Voice control not available. Set BOSON_API_KEY environment variable.")
            return False
        
        self.voice_enabled = not self.voice_enabled
        status = "enabled" if self.voice_enabled else "disabled"
        print(f"🎤 Voice control {status}")
        
        # Send status update to web interface
        self.send_command({
            "type": "voice_status",
            "enabled": self.voice_enabled,
            "status": status
        })
        
        if self.voice_enabled:
            self.show_voice_help()
        return True
    
    def force_kill(self):
        """Forcefully terminate the controller and all processes."""
        print("💀 Force killing all processes...")
        try:
            # Close WebSocket connection
            if self.ws:
                self.ws.close()
            
            # Set running to False
            self.running = False
            
            # Force exit the process
            import sys
            import os
            print("💀 Terminating process...")
            os._exit(0)  # Force exit without cleanup
            
        except Exception as e:
            print(f"💀 Kill error: {e}")
            # Last resort - force exit
            import os
            os._exit(1)
    
    def process_image_editing_command(self, command_text):
        """Process a voice command as an image editing instruction."""
        try:
            # Set processing flag to prevent new voice commands
            self.processing_edit = True
            self.processing_start_time = time.time()
            
            # Get LM output for the command
            edits = get_lm_output(command_text)
            if not edits or len(edits) == 0:
                print("No editing instructions detected.")
                self.processing_edit = False  # Reset flag
                self.processing_start_time = None
                return False
            
            print(f"Received {len(edits)} editing instructions.")
            
            # Store edits and set pending flag
            self._pending_edits = edits
            self._editing_pending = True
            
            # Get current image
            if not self.send_command({"type": "get_current_image"}):
                self.processing_edit = False  # Reset flag
                self.processing_start_time = None
                return False
            
            # Wait for response
            time.sleep(0.1)
            return True
            
        except Exception as e:
            print(f"✗ Error processing image editing command: {e}")
            self.processing_edit = False  # Reset flag
            return False
    
    def create_backup(self, image_name):
        """Create a backup of the image, reusing deleted numbers when possible."""
        try:
            test_images_path = './test_images'
            image_path = os.path.join(test_images_path, image_name)
            
            if not os.path.exists(image_path):
                print(f"✗ Image '{image_name}' not found")
                return False
            
            base_name = os.path.splitext(image_name)[0]
            ext = os.path.splitext(image_name)[1]
            
            # Find all existing backup numbers
            existing_numbers = set()
            highest_number = 0
            
            # Look for existing backup files
            for file in os.listdir(test_images_path):
                if file.startswith(f"{base_name}_old") and file.endswith(ext):
                    try:
                        # Extract number from filename like "image_old1.png" -> 1
                        number_part = file[len(f"{base_name}_old"):-len(ext)]
                        if number_part.isdigit():
                            number = int(number_part)
                            existing_numbers.add(number)
                            highest_number = max(highest_number, number)
                    except ValueError:
                        continue
            
            # Find the next available number (reuse gaps if possible)
            next_number = 1
            while next_number in existing_numbers:
                next_number += 1
            
            # If no gaps, use the next number after highest
            if next_number <= highest_number:
                next_number = highest_number + 1
            
            backup_path = os.path.join(test_images_path, f"{base_name}_old{next_number}{ext}")
            
            with Image.open(image_path) as backup_img:
                backup_img.save(backup_path)
                print(f"✓ Created backup: {base_name}_old{next_number}{ext}")
            
            # Update undo count (total backups available)
            self.image_undo_counts[image_name] = next_number
            
            return True
            
        except Exception as e:
            print(f"✗ Error creating backup: {e}")
            return False

    def process_image_edits(self, image_name, edits):
        """Process a list of image edits on the current image."""
        try:
            test_images_path = './test_images'
            image_path = os.path.join(test_images_path, image_name)
            
            if not os.path.exists(image_path):
                print(f"✗ Image '{image_name}' not found")
                return False
            
            # Create backup before editing
            if not self.create_backup(image_name):
                print("⚠️ Failed to create backup, proceeding with edits anyway")
            
            # Process each edit sequentially
            for edit in edits:
                print(f"Processing edit: {edit}")
                
                if edit.get('generative'):
                    # Process generative edit using GPU server
                    prompt = edit.get('prompt', '')
                    if prompt:
                        success = process_generative_edit(image_path, prompt)
                        if not success:
                            print(f"⚠️ Generative edit failed: '{prompt}'")
                    else:
                        print("⚠️ No prompt provided for generative edit")
                    continue
                
                # Handle native edits
                action_str = edit.get('action')
                if not action_str or not isinstance(action_str, str):
                    print(f"Invalid action format: {action_str}")
                    continue
                
                # Parse action string e.g., '(rotate, 90)' -> 'rotate', ['90']
                try:
                    cleaned_str = action_str.strip().strip('()')
                    parts = [p.strip() for p in cleaned_str.split(',')]
                    action_name = parts[0]
                    params = [p for p in parts[1:] if p]  # Filter out empty strings
                    
                    # Load current image for native edit
                    with Image.open(image_path) as img:
                        if action_name == 'vflip':
                            img = vflip(img)
                            print("Applied 'vflip' successfully.")
                        elif action_name == 'hflip':
                            img = hflip(img)
                            print("Applied 'hflip' successfully.")
                        elif action_name == 'grayscale':
                            img = grayscale(img)
                            print("Applied 'grayscale' successfully.")
                        elif action_name == 'rotate':
                            if len(params) == 1:
                                degrees = float(params[0])
                                img = rotate(img, degrees=degrees)
                                print(f"Applied 'rotate' with {degrees} degrees.")
                            else:
                                print("Error: 'rotate' requires one parameter (degrees).")
                                continue
                        elif action_name == 'saturation':
                            if len(params) == 1:
                                factor = float(params[0])
                                img = saturation(img, factor=factor)
                                print(f"Applied 'saturation' with factor {factor}.")
                            else:
                                print("Error: 'saturation' requires one parameter (factor).")
                                continue
                        elif action_name == 'sharpness':
                            if len(params) == 1:
                                factor = float(params[0])
                                img = sharpness(img, factor=factor)
                                print(f"Applied 'sharpness' with factor {factor}.")
                            else:
                                print("Error: 'sharpness' requires one parameter (factor).")
                                continue
                        elif action_name == 'resize':
                            if len(params) == 2:
                                try:
                                    # Extract numbers from the potentially messy parameter strings
                                    width = int(re.search(r'\d+', params[0]).group())
                                    height = int(re.search(r'\d+', params[1]).group())
                                    size = (width, height)
                                    img = resize(img, size=size)
                                    print(f"Applied 'resize' with size {size}.")
                                except (AttributeError, ValueError):
                                    print(f"Error: could not parse width/height from resize parameters: {params}")
                                    continue
                            else:
                                print("Error: 'resize' requires two parameters (width, height).")
                                continue
                        else:
                            print(f"Unknown action: '{action_name}'")
                            continue
                        
                        # Save the native edit immediately
                        img.save(image_path)
                        print(f"✓ Saved native edit: {action_name}")
                
                except ValueError as e:
                    print(f"Error converting parameter for '{action_name}': {e}")
                except Exception as e:
                    print(f"Error applying edit '{action_name}': {e}")
            
            print(f"✓ All edits completed for: {image_name}")
            
            # Clear processing flag to allow new voice commands
            self.processing_edit = False
            self.processing_start_time = None
            return True
                
        except Exception as e:
            print(f"✗ Error processing image edits: {e}")
            # Clear processing flag even on error
            self.processing_edit = False
            self.processing_start_time = None
            return False
    
    def find_and_select_image(self, search_term):
        """Find an image by partial name match and select it."""
        try:
            test_images_path = './test_images'
            
            if not os.path.exists(test_images_path):
                print("✗ Images directory not found")
                return False
            
            # Get all image files
            files = os.listdir(test_images_path)
            image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg']
            
            # Filter for images and exclude _old files
            images = []
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in image_extensions and '_old' not in file:
                    images.append(file)
            
            if not images:
                print("✗ No images found in directory")
                return False
            
            # Search for partial matches (case insensitive)
            search_term_lower = search_term.lower()
            matches = []
            
            for image in images:
                if search_term_lower in image.lower():
                    matches.append(image)
            
            if not matches:
                print(f"✗ No images found containing '{search_term}'")
                print("Available images:")
                for image in images[:10]:  # Show first 10 images
                    print(f"  - {image}")
                if len(images) > 10:
                    print(f"  ... and {len(images) - 10} more")
                return False
            
            if len(matches) == 1:
                # Single match - select it
                selected_image = matches[0]
                print(f"✓ Found: {selected_image}")
                return self.select_image(selected_image)
            else:
                # Multiple matches - show options
                print(f"Found {len(matches)} images containing '{search_term}':")
                for i, image in enumerate(matches, 1):
                    print(f"  {i}. {image}")
                print("Please be more specific or use the exact filename")
                return False
            
        except Exception as e:
            print(f"✗ Error finding image: {e}")
            return False
    
    def run(self):
        """Main terminal loop"""
        if not self.connect():
            return
        
        print("\n🎮 Image Viewer Terminal Controller")
        print("Commands:")
        print("  ls               - List available images")
        print("  current          - Show currently selected image")
        print("  revert           - Restore image from backup (unlimited)")
        print("  revert X steps   - Multi-step revert (X = number of steps)")
        print("  status           - Show undo status for current image")
        print("  open <filename>  - Select an image")
        print("  voice            - Toggle voice control on/off")
        print("  quit/exit        - Exit the controller")
        print("  help             - Show this help")
        print()
        print("📊 Smart Backup System:")
        print("  - Each edit creates a backup (image_old1, image_old2, etc.)")
        print("  - Use 'revert' to undo the most recent edit (deletes that backup)")
        print("  - New edits reuse deleted backup numbers")
        print("  - Highest number = most recent backup")
        print("  - Use 'status' to see all available backups")
        print()
        
        if self.boson_client:
            print("🎤 Voice control available! Say 'voice' to enable or 'help' to see voice commands.")
        else:
            print("⚠️  Voice control disabled. Set BOSON_API_KEY environment variable to enable.")
        print()
        
        while self.running and self.connected:
            try:
                if self.voice_enabled:
                    # Check if we're processing an edit
                    if self.processing_edit:
                        # Check for timeout (90 seconds max processing time for generative edits)
                        if self.processing_start_time and time.time() - self.processing_start_time > 90:
                            print("⏱️ Processing timeout - resetting voice control")
                            self.processing_edit = False
                            self.processing_start_time = None
                        else:
                            print("🎨 Processing image edit... Please wait.")
                            time.sleep(1)  # Wait a bit before checking again
                            continue
                    
                    # Voice control mode
                    print("🎤 Listening for voice command...")
                    # Send listening status
                    self.send_command({
                        "type": "voice_status",
                        "enabled": True,
                        "status": "listening"
                    })
                    
                    audio_b64 = self.record_until_silence()
                    if audio_b64:
                        # Send processing status
                        self.send_command({
                            "type": "voice_status",
                            "enabled": True,
                            "status": "processing"
                        })
                        
                        command_text = self.transcribe_audio(audio_b64)
                        if command_text:
                            self.process_voice_command(command_text)
                        else:
                            print("⚠️ Could not transcribe audio, trying again...")
                    else:
                        print("⚠️ No audio captured, trying again...")
                else:
                    # Keyboard input mode
                    command = input("> ").strip()
                    
                    if not command:
                        continue
                    
                    if command.lower() in ['quit', 'exit']:
                        print("👋 Goodbye!")
                        break
                    elif command.lower() in ['kill', 'force quit', 'terminate', 'die']:
                        print("💀 Force killing controller...")
                        self.force_kill()
                        break
                    elif command.lower() == 'help':
                        print("Commands:")
                        print("  ls               - List available images")
                        print("  current          - Show currently selected image")
                        print("  revert           - Restore image from backup (unlimited)")
                        print("  status           - Show undo status for current image")
                        print("  open <filename>  - Select an image")
                        print("  voice            - Toggle voice control on/off")
                        print("  quit/exit        - Exit the controller")
                        print("  kill             - Force kill the controller")
                        print("  help             - Show this help")
                        print()
                        print("📊 Smart Backup System:")
                        print("  - Each edit creates a backup (image_old1, image_old2, etc.)")
                        print("  - Use 'revert' to undo the most recent edit (deletes that backup)")
                        print("  - New edits reuse deleted backup numbers")
                        print("  - Highest number = most recent backup")
                        print("  - Use 'status' to see all available backups")
                    elif command.lower() == 'voice':
                        self.toggle_voice_control()
                    elif command.lower() == 'ls':
                        self.list_images()
                    elif command.lower() == 'current':
                        self.get_current_image()
                    elif command.lower() in ['undo', 'revert']:
                        self.undo_image()
                    elif command.startswith('revert ') and 'step' in command:
                        # Parse multi-step revert command
                        try:
                            # Extract number from command like "revert 4 steps"
                            import re
                            match = re.search(r'revert\s+(\d+)\s+steps?', command.lower())
                            if match:
                                steps = int(match.group(1))
                                print(f"🎯 Multi-step revert: {steps} steps")
                                self.undo_multiple_steps(steps)
                            else:
                                print("✗ Invalid format. Use: revert X steps")
                        except ValueError:
                            print("✗ Invalid number. Use: revert X steps")
                    elif command.lower() in ['status', 'undo status']:
                        if hasattr(self, '_current_image_name') and self._current_image_name:
                            self.show_undo_status(self._current_image_name)
                        else:
                            print("✗ No image currently selected")
                    elif command.startswith('open '):
                        image_name = command[5:].strip()
                        if image_name:
                            self.select_image(image_name)
                        else:
                            print("✗ Please specify an image name")
                    else:
                        print("✗ Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except EOFError:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"✗ Unexpected error: {e}")
                print("Continuing... (Press Ctrl+C to exit)")
                continue
        
        self.running = False
        if self.ws:
            self.ws.close()

def main():
    controller = ImageViewerController()
    controller.run()

if __name__ == "__main__":
    main()
