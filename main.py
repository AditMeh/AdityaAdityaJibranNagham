import threading
import time
from PIL import Image
import sys
import re

# Import your custom modules
from jibran.lmoutput import get_lm_output
from Nagham.liveAudioTranscribing import record_until_silence, analyze_audio

# Import all native image editing functions
from jibran.native import (
    grayscale,
    hflip,
    resize,
    rotate,
    saturation,
    sharpness,
    vflip,
)

def process_image_with_voice(image_path: str):
    """
    Loads an image and allows real-time editing through voice commands.
    """
    try:
        img = Image.open(image_path)
        print(f"Image '{image_path}' loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Image not found at '{image_path}'")
        return

    # Show the initial image
    img.show(title="Original Image")

    def audio_listener_loop():
        nonlocal img
        while True:
            audio_b64 = record_until_silence()
            if not audio_b64:
                continue

            transcription = analyze_audio(audio_b64)
            if not transcription:
                continue

            print(f"Transcribed text: {transcription}")

            edits = get_lm_output(transcription)
            if not edits or len(edits) == 0:
                print("No instructions detected.")
                continue
            
            print(f"Received {len(edits)} edits.")
            for edit in edits:
                print(f"Processing edit: {edit}")

                if edit.get('generative'):
                    print("Generative edit is not yet implemented.")
                    """ Fill in the generative part here""" # TODO: Implement the generative part
                    continue

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

                    if action_name == 'vflip':
                        img = vflip.vflip(img)
                        print("Applied 'vflip' successfully.")
                        img.show(title="After: vflip")
                    elif action_name == 'hflip':
                        img = hflip.hflip(img)
                        print("Applied 'hflip' successfully.")
                        img.show(title="After: hflip")
                    elif action_name == 'grayscale':
                        img = grayscale.grayscale(img)
                        print("Applied 'grayscale' successfully.")
                        img.show(title="After: grayscale")
                    elif action_name == 'rotate':
                        if len(params) == 1:
                            degrees = float(params[0])
                            img = rotate.rotate(img, degrees=degrees)
                            print(f"Applied 'rotate' with {degrees} degrees.")
                            img.show(title=f"After: rotate {degrees}")
                        else:
                            print("Error: 'rotate' requires one parameter (degrees).")
                    elif action_name == 'saturation':
                        if len(params) == 1:
                            factor = float(params[0])
                            img = saturation.saturation(img, factor=factor)
                            print(f"Applied 'saturation' with factor {factor}.")
                            img.show(title=f"After: saturation {factor}")
                        else:
                            print("Error: 'saturation' requires one parameter (factor).")
                    elif action_name == 'sharpness':
                        if len(params) == 1:
                            factor = float(params[0])
                            img = sharpness.sharpness(img, factor=factor)
                            print(f"Applied 'sharpness' with factor {factor}.")
                            img.show(title=f"After: sharpness {factor}")
                        else:
                            print("Error: 'sharpness' requires one parameter (factor).")
                    elif action_name == 'resize':
                        if len(params) == 2:
                            try:
                                # Extract numbers from the potentially messy parameter strings
                                width = int(re.search(r'\d+', params[0]).group())
                                height = int(re.search(r'\d+', params[1]).group())
                                size = (width, height)
                                img = resize.resize(img, size=size)
                                print(f"Applied 'resize' with size {size}.")
                                img.show(title=f"After: resize {size}")
                            except (AttributeError, ValueError):
                                print(f"Error: could not parse width/height from resize parameters: {params}")
                        else:
                            print("Error: 'resize' requires two parameters (width, height).")
                    else:
                        print(f"Unknown action: '{action_name}'")

                except ValueError as e:
                    print(f"Error converting parameter for '{action_name}': {e}")
                except Exception as e:
                    print(f"Error applying edit '{action_name}': {e}")


    # === Start Listener Thread ===
    listener_thread = threading.Thread(target=audio_listener_loop, daemon=True)
    listener_thread.start()

    print("Live audio listener started. Press Ctrl+C to save and exit.")
    try:
        # Keep the main thread alive to listen for KeyboardInterrupt
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting gracefully.")
        final_path = "edited_image.png"
        img.save(final_path)
        print(f"Final image saved to '{final_path}'")


# === MAIN EXECUTION ===
if __name__ == "__main__":
    # If a path is provided as a command-line argument, use it.
    # Otherwise, use the default path.
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "example.jpg"
    
    process_image_with_voice(image_path)