import torch
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from transformers import Qwen2_5_VLForConditionalGeneration

from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel
from PIL import Image
import sys
import threading
import os
import random
import time

model_id = "Qwen/Qwen-Image-Edit-2509"
torch_dtype = torch.bfloat16
device = "cuda"

# -------- Model Initialization (startup, happens ONCE) --------
def init_models():
    diffusers_quant_config = DiffusersBitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=["transformer_blocks.0.img_mod"],
    )
    transformer = QwenImageTransformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        quantization_config=diffusers_quant_config,
        torch_dtype=torch_dtype,
        device_map=device,
    )

    transformers_quant_config = TransformersBitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        subfolder="text_encoder",
        quantization_config=transformers_quant_config,
        torch_dtype=torch_dtype,
        device_map=device,
    )

    pipe = QwenImageEditPlusPipeline.from_pretrained(
        model_id, transformer=transformer, text_encoder=text_encoder, torch_dtype=torch_dtype
    )
    
    # Load LoRA weights to speed up inference
    pipe.load_lora_weights("lightx2v/Qwen-Image-Lightning", weight_name="Qwen-Image-Lightning-8steps-V1.1.safetensors")
    pipe.load_lora_weights(
        "lightx2v/Qwen-Image-Lightning", weight_name="Qwen-Image-Lightning-4steps-V1.0-bf16.safetensors"
    )
    
    # Enable CPU offload to manage memory
    pipe.enable_model_cpu_offload()
    pipe.to(device)
    return pipe

pipe = init_models()

DEFAULT_SEED = 42

def create_generator(seed=None):
    global device
    if seed is None:
        # Generate a fresh, high-entropy random seed
        seed = random.SystemRandom().randint(0, 2**63 - 1)
    return torch.Generator(device=device).manual_seed(seed), seed

generator, current_seed = create_generator(DEFAULT_SEED)

print("Multi-image inference server is ready.")
print("Usage: Enter image paths separated by commas, then a pipe, then the prompt:")
print("Example: murdock.jpeg,frog.jpg|The man is holding the frog")
print("To quit, enter an empty line or Ctrl-C.\n")
print("Commands:")
print("  :seed [N]     - Set a manual seed for deterministic results (e.g., :seed 12345)")
print("  :seed random  - Use a random seed for novel results")

def inference_loop():
    global generator, current_seed
    while True:
        try:
            user_input = input("IMAGE_PATHS|PROMPT or :seed [N/random]: ").strip()
            if not user_input:
                print("Exiting inference server.")
                break

            # Allow seed change from terminal
            if user_input.startswith(":seed"):
                parts = user_input.split()
                if len(parts) == 2:
                    if parts[1] == "random":
                        generator, current_seed = create_generator(None)
                        print(f"Randomized seed, using seed: {current_seed}")
                    else:
                        try:
                            seed = int(parts[1])
                            generator, current_seed = create_generator(seed)
                            print(f"Set seed to {seed}")
                        except ValueError:
                            print("Invalid seed value. Must be an integer or 'random'.")
                else:
                    print("Usage: :seed N or :seed random")
                continue

            if "|" not in user_input:
                print("Please use format: IMAGE_PATHS|PROMPT (separate multiple images with commas)")
                continue
            
            image_paths_str, prompt = user_input.split("|", 1)
            image_paths_str = image_paths_str.strip()
            prompt = prompt.strip()
            
            # Parse multiple image paths
            image_paths = [path.strip() for path in image_paths_str.split(",")]
            
            # Validate all image paths exist
            missing_images = []
            for path in image_paths:
                if not os.path.exists(path):
                    missing_images.append(path)
            
            if missing_images:
                print(f"Missing images: {', '.join(missing_images)}")
                continue
                
            if not prompt:
                print("Prompt cannot be empty.")
                continue
                
            try:
                # Load all images
                images = []
                for path in image_paths:
                    image = Image.open(path).convert("RGB")
                    images.append(image)
                print(f"Loaded {len(images)} images")
            except Exception as e:
                print(f"Failed to open images: {e}")
                continue

            # Run inference
            print(f"Running multi-image generation for: '{prompt}' (seed={current_seed})...")
            
            inputs = {
                "image": images,
                "prompt": prompt,
                "generator": generator,
                "true_cfg_scale": 4.0,
                "negative_prompt": "out of frame, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature",
                "num_inference_steps": 8,
                "guidance_scale": 1.0,
                "num_images_per_prompt": 1,
            }
            
            with torch.inference_mode():
                output = pipe(**inputs)
                output_image = output.images[0]
                
                # Generate output filename based on first input image
                base_name = os.path.splitext(os.path.basename(image_paths[0]))[0]
                output_path = f"{base_name}_multi_edit.png"
                output_image.save(output_path)
                print(f"Saved generated image to: {os.path.abspath(output_path)}")
                
        except KeyboardInterrupt:
            print("\nExiting inference server.")
            break
        except Exception as ex:
            print(f"Error: {ex}")

if __name__ == "__main__":
    inference_loop()
