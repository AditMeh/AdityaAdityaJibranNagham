import torch
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from transformers import Qwen2_5_VLForConditionalGeneration

from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers import QwenImageEditPipeline, QwenImageTransformer2DModel
from PIL import Image
import sys
import threading
import os
import random
import time

model_id = "Qwen/Qwen-Image-Edit"
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

    pipe = QwenImageEditPipeline.from_pretrained(
        model_id, transformer=transformer, text_encoder=text_encoder, torch_dtype=torch_dtype
    )
    pipe.load_lora_weights("lightx2v/Qwen-Image-Lightning", weight_name="Qwen-Image-Lightning-8steps-V1.1.safetensors")
    # pipe.load_lora_weights("lightx2v/Qwen-Image-Lightning", weight_name="Qwen-Image-Lightning-4steps-V1.0-bf16.safetensors")
    # Inference will use GPU as much as possible. Remove CPU offload to keep everything in VRAM.
    # pipe.enable_model_cpu_offload()  # Not needed if everything is on cuda
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

print("Inference server is ready.")
print("Usage: After the prompt, enter the image path and prompt for each job:")
print("Example: murdock.jpeg|Make the person bald, give him a beard and change the background to a forest background")
print("To quit, enter an empty line or Ctrl-C.\n")
print("Commands:")
print("  :seed [N]     - Set a manual seed for deterministic results (e.g., :seed 12345)")
print("  :seed random  - Use a random seed for novel results")

def inference_loop():
    global generator, current_seed
    while True:
        try:
            user_input = input("IMAGE_PATH|PROMPT or :seed [N/random]: ").strip()
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
                print("Please use format: IMAGE_PATH|PROMPT")
                continue
            image_path, prompt = user_input.split("|", 1)
            image_path = image_path.strip()
            prompt = prompt.strip()
            if not (os.path.exists(image_path) and prompt):
                print(f"Image '{image_path}' does not exist or prompt is empty.")
                continue
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Failed to open image: {e}")
                continue

            # Run inference
            print(f"Running image generation for: '{prompt}' (seed={current_seed})...")
            output = pipe(image, prompt, num_inference_steps=6, generator=generator, num_images_per_prompt=2, negative_prompt="")
            print(len(output.images))
            for idx, out_img in enumerate(output.images):
                output_path = f"{os.path.splitext(image_path)[0]}_edit_{idx+1}.png"
                out_img.save(output_path)
                print(f"Saved generated image {idx+1} to: {output_path}")
        except KeyboardInterrupt:
            print("\nExiting inference server.")
            break
        except Exception as ex:
            print(f"Error: {ex}")

if __name__ == "__main__":
    inference_loop()
